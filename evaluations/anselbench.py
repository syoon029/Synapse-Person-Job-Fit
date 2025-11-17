import json
import numpy as np
import pandas as pd
import faiss
from infrastructure.database import Posting, Resume
from embeddings.embed import embed_posting, embed_resume_text, _safe_join
from embeddings.embed_stage2 import embed_text, doc_sim_score
from typing import List, Dict, Any
from scipy.stats import rankdata

from sklearn.metrics import ndcg_score
from sklearn.preprocessing import minmax_scale, StandardScaler

from llm_scorer import load_cache, pairwise_rank_to_scores

def read_ansel_resume(file_path='resume_data/ansel_resume.csv'):
    df = pd.read_csv(file_path)
    for row in df.itertuples():
        ansel_resume = Resume(id=1, text=row.Resume_str)
    return ansel_resume

def read_ansel_postings(file_path='linkedin_data/ansel_postings.csv'):
    df = pd.read_csv(file_path)
    postings = df.to_dict(orient='records')
    all_postings = []
    for posting in postings:
        posting_obj = Posting(
            title=posting['posting.title'],
            company_name=posting['posting.company_name'],
            skills_desc=posting['posting.skills_desc'],
            description=posting['posting.description'],
            formatted_experience_level=posting['posting.formatted_experience_level'],
            location=posting['posting.location']
        )
        all_postings.append(posting_obj)
    return all_postings

def initialize_faiss_index_for_ansel(postings):
    ids = np.array([i for i in range(len(postings))], dtype='int64')
    embeddings = np.array([np.array(json.loads(embed_posting(posting)), dtype='float32') for posting in postings])
    emb_dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(emb_dim))
    index.add_with_ids(embeddings, ids)
    return index

def companies_that_interviewed_ansel():
    interviews = ["Baseten", "Google", "Palantir Technologies", "Databricks", "Headlands Tech", "Zip", "TikTok", "Applied Intuition", "Cape", "Optiver"]
    offers = ["Google", "Applied Intuition", "Baseten", "Palantir Technologies", "Databricks"]
    return interviews, offers

def get_relevance(posting: Posting, offers_set: set, interviews_set: set) -> int:
    """Assigns relevance score based on ground truth lists."""
    if posting.company_name in offers_set:
        return 2  # Most relevant
    if posting.company_name in interviews_set:
        return 1  # Relevant
    return 0  # Not relevant

# --- NEW: Advanced Normalization Functions ---

def normalize_scores_minmax(scores_np):
    """Min-max normalization to [0, 1]"""
    scores_no_inf = scores_np.copy()
    scores_no_inf[scores_no_inf == -np.inf] = -999.0
    return minmax_scale(scores_no_inf, axis=1)

def normalize_scores_zscore(scores_np):
    """Z-score normalization (standardization)"""
    scores_no_inf = scores_np.copy()
    scores_no_inf[scores_no_inf == -np.inf] = np.nanmin(scores_no_inf[scores_no_inf != -np.inf]) - 1
    scaler = StandardScaler()
    normalized = scaler.fit_transform(scores_no_inf.T).T
    # Shift to positive range and scale to [0, 1]
    return minmax_scale(normalized, axis=1)

def normalize_scores_rank(scores_np):
    """Rank-based normalization (convert to percentile ranks)"""
    scores_flat = scores_np.flatten()
    # Handle -inf by replacing with very low value
    scores_flat[scores_flat == -np.inf] = -999.0
    # Get ranks (higher score = higher rank)
    ranks = rankdata(scores_flat, method='average')
    # Normalize to [0, 1]
    normalized = (ranks - 1) / (len(ranks) - 1)
    return normalized.reshape(scores_np.shape)

def normalize_scores_softmax(scores_np, temperature=1.0):
    """Softmax normalization with temperature control"""
    scores_no_inf = scores_np.copy()
    scores_no_inf[scores_no_inf == -np.inf] = -999.0
    # Apply temperature scaling
    scores_scaled = scores_no_inf / temperature
    # Subtract max for numerical stability
    scores_shifted = scores_scaled - np.max(scores_scaled, axis=1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# --- NEW: Advanced Ensemble Functions ---

def reciprocal_rank_fusion(ranks_list, k=60):
    """
    Reciprocal Rank Fusion: Combines multiple rankings by summing 1/(k+rank)
    This is scale-invariant and robust to outliers.
    
    Args:
        ranks_list: List of rank arrays (1-indexed)
        k: Constant to prevent division by zero and reduce impact of top ranks
    """
    n = len(ranks_list[0])
    fused_scores = np.zeros(n)
    
    for ranks in ranks_list:
        # Convert to 0-indexed, handle N/A ranks
        ranks_array = np.array(ranks, dtype=float)
        ranks_array[ranks_array < 0] = n + 1  # Put N/A at the end
        fused_scores += 1.0 / (k + ranks_array)
    
    return fused_scores / len(ranks_list)  # Average

def borda_count(ranks_list):
    """
    Borda Count: Assigns points based on position (n-rank points)
    Higher is better.
    """
    n = len(ranks_list[0])
    fused_scores = np.zeros(n)
    
    for ranks in ranks_list:
        ranks_array = np.array(ranks, dtype=float)
        ranks_array[ranks_array < 0] = n + 1  # Put N/A at the end
        fused_scores += (n + 1 - ranks_array)
    
    return fused_scores / len(ranks_list)

def weighted_geometric_mean(scores_list, weights):
    """
    Geometric mean with weights: product of (score^weight)
    Good for requiring consensus across rankers.
    """
    result = np.ones_like(scores_list[0])
    for score, weight in zip(scores_list, weights):
        # Add small epsilon to avoid log(0)
        result *= np.power(score + 1e-10, weight)
    return result

def weighted_harmonic_mean(scores_list, weights):
    """
    Harmonic mean with weights: More sensitive to low scores
    Good when you want all rankers to agree.
    """
    result = np.zeros_like(scores_list[0])
    total_weight = sum(weights)
    
    for score, weight in zip(scores_list, weights):
        result += weight / (score + 1e-10)
    
    return total_weight / result

if __name__ == "__main__":
    print("Loading Data...")
    ansel_resume = read_ansel_resume()
    ansel_postings = read_ansel_postings()
    interviews, offers = companies_that_interviewed_ansel()

    interviews_set = set(interviews)
    offers_set = set(offers)

    print("Initializing FAISS Index...")
    faiss_index = initialize_faiss_index_for_ansel(ansel_postings)
    
    # PHASE 1 RECOMMENDATIONS
    print("Starting Phase 1 Recommendations...")
    phase1_resume_embedding = embed_resume_text(ansel_resume.text)
    scores, ids = faiss_index.search(phase1_resume_embedding.reshape(1, -1), 100)
    phase1_scores, phase1_ids = scores[0], ids[0]

    phase1_scores_all = np.full(len(ansel_postings), -np.inf, dtype=np.float32)
    for i in range(len(phase1_ids)):
        idx = phase1_ids[i]
        if idx >= 0:
            phase1_scores_all[idx] = phase1_scores[i]
    phase1_scores_all_np = np.array([phase1_scores_all])
    
    # Create Phase 1 ranks
    phase1_ranks = np.full(len(ansel_postings), -1, dtype=int)
    for rank, idx in enumerate(phase1_ids):
        if idx >= 0:
            phase1_ranks[idx] = rank + 1

    # PHASE 2 RECOMMENDATIONS
    print("Starting Phase 2 Recommendations...")
    scored_postings = []
    phase2_scores_all = []
    
    phase2_resume_embedding = embed_text(ansel_resume.text)
    for i, posting in enumerate(ansel_postings):
        posting_embedding = embed_text(_safe_join([
            posting.title, posting.company_name, posting.skills_desc,
            posting.description, posting.formatted_experience_level, posting.location
        ]))
        
        score = doc_sim_score(phase2_resume_embedding, posting_embedding)
        scored_postings.append((score, posting))
        phase2_scores_all.append(score)

    scored_postings.sort(reverse=True, key=lambda x: x[0])
    phase2_scores_all_np = np.array([phase2_scores_all])
    
    # Create Phase 2 ranks
    phase2_ranks = np.empty(len(ansel_postings), dtype=int)
    for rank, (score, posting) in enumerate(scored_postings, start=1):
        original_index = ansel_postings.index(posting)
        phase2_ranks[original_index] = rank
    
    # PHASE 3 (LLM) PAIRWISE RANKING
    print("\nStarting Phase 3 (LLM) Pairwise Ranking...")
    
    llm_cache = load_cache()
    print(f"Loaded {len(llm_cache)} items from LLM cache.")
    
    postings_texts = []
    for posting in ansel_postings:
        posting_text = _safe_join([
            f"Title: {posting.title}",
            f"Company: {posting.company_name}",
            f"Location: {posting.location}",
            f"Experience Level: {posting.formatted_experience_level}",
            f"Skills: {posting.skills_desc}",
            f"Description: {posting.description}"
        ])
        postings_texts.append(posting_text)
    
    llm_scores_all = pairwise_rank_to_scores(
        resume_text=ansel_resume.text,
        postings_texts=postings_texts,
        cache=llm_cache
    )
    
    llm_ranked_indices = np.argsort(llm_scores_all)[::-1]
    llm_ranks = np.empty(len(llm_scores_all), dtype=int)
    for rank, idx in enumerate(llm_ranked_indices):
        llm_ranks[idx] = rank + 1
    
    llm_scores_all_np = np.array([llm_scores_all])
    print("LLM Pairwise Ranking complete.")

    # --- ADVANCED ENSEMBLE CALCULATIONS ---
    print("\nCalculating Advanced Ensemble Scores...")
    
    # 1. Try different normalization techniques
    norm_methods = {
        'minmax': (normalize_scores_minmax, {}),
        'zscore': (normalize_scores_zscore, {}),
        'rank': (normalize_scores_rank, {}),
        'softmax_t0.5': (normalize_scores_softmax, {'temperature': 0.5}),
        'softmax_t1.0': (normalize_scores_softmax, {'temperature': 1.0}),
    }
    
    ensemble_configs = {}
    
    # Test each normalization method with simple weighted average
    for norm_name, (norm_func, norm_kwargs) in norm_methods.items():
        p1_norm = norm_func(phase1_scores_all_np, **norm_kwargs)
        p2_norm = norm_func(phase2_scores_all_np, **norm_kwargs)
        llm_norm = norm_func(llm_scores_all_np, **norm_kwargs)
        
        # Ensure all are 2D (1, n) arrays
        p1_norm = p1_norm.reshape(1, -1) if p1_norm.ndim == 1 else p1_norm
        p2_norm = p2_norm.reshape(1, -1) if p2_norm.ndim == 1 else p2_norm
        llm_norm = llm_norm.reshape(1, -1) if llm_norm.ndim == 1 else llm_norm
        
        # Weighted average (P1 dominant since it's best)
        ensemble_configs[f'WAvg_{norm_name}_90-5-5'] = 0.90 * p1_norm + 0.05 * p2_norm + 0.05 * llm_norm
        ensemble_configs[f'WAvg_{norm_name}_80-10-10'] = 0.80 * p1_norm + 0.10 * p2_norm + 0.10 * llm_norm
        ensemble_configs[f'WAvg_{norm_name}_70-15-15'] = 0.70 * p1_norm + 0.15 * p2_norm + 0.15 * llm_norm
    
    # 2. Rank-based fusion methods (scale-invariant)
    rrf_scores = reciprocal_rank_fusion([phase1_ranks, phase2_ranks, llm_ranks], k=60)
    ensemble_configs['RRF_k60'] = np.array([rrf_scores])
    
    rrf_scores_k30 = reciprocal_rank_fusion([phase1_ranks, phase2_ranks, llm_ranks], k=30)
    ensemble_configs['RRF_k30'] = np.array([rrf_scores_k30])
    
    borda_scores = borda_count([phase1_ranks, phase2_ranks, llm_ranks])
    ensemble_configs['Borda'] = np.array([borda_scores])
    
    # 3. Non-linear combinations (using minmax normalized scores)
    p1_minmax = normalize_scores_minmax(phase1_scores_all_np)
    p2_minmax = normalize_scores_minmax(phase2_scores_all_np)
    llm_minmax = normalize_scores_minmax(llm_scores_all_np)
    
    # Ensure all are properly shaped
    p1_minmax = p1_minmax.reshape(1, -1)
    p2_minmax = p2_minmax.reshape(1, -1)
    llm_minmax = llm_minmax.reshape(1, -1)
    
    # Geometric mean (requires consensus)
    geom_scores = weighted_geometric_mean([p1_minmax.flatten(), p2_minmax.flatten(), llm_minmax.flatten()], [0.8, 0.1, 0.1])
    ensemble_configs['GeomMean_80-10-10'] = np.array([geom_scores])
    
    # Harmonic mean (very sensitive to disagreement)
    harm_scores = weighted_harmonic_mean([p1_minmax.flatten(), p2_minmax.flatten(), llm_minmax.flatten()], [0.8, 0.1, 0.1])
    ensemble_configs['HarmMean_80-10-10'] = np.array([harm_scores])
    
    # Max fusion (optimistic - take best signal)
    max_scores = np.maximum(np.maximum(p1_minmax, p2_minmax), llm_minmax)
    ensemble_configs['MaxFusion'] = max_scores
    
    # Hybrid: P1 with LLM boost only at top-k
    hybrid_scores = p1_minmax.copy()
    llm_rank_array = np.array([llm_ranks])
    top_llm_mask = (llm_rank_array <= 20)  # Boost if in LLM's top 20
    hybrid_scores[top_llm_mask] += 0.1 * llm_minmax[top_llm_mask]
    ensemble_configs['P1_LLMBoost_top20'] = hybrid_scores
    
    print(f"Created {len(ensemble_configs)} ensemble configurations.")

    # --- nDCG EVALUATION ---
    print("\nCalculating nDCG for all methods...")
    
    true_relevances = np.array([[
        get_relevance(p, offers_set, interviews_set) for p in ansel_postings
    ]])
    
    k_values = [10, 20, 50, 100]
    ndcg_results = {}
    
    # Baseline methods
    baseline_methods = {
        'Phase 1': phase1_scores_all_np,
        'Phase 2': phase2_scores_all_np,
        'LLM': llm_scores_all_np,
    }
    
    all_methods = {**baseline_methods, **ensemble_configs}
    
    for k in k_values:
        ndcg_results[f"nDCG@{k}"] = {}
        
        for method_name, scores in all_methods.items():
            ndcg = ndcg_score(true_relevances, scores, k=k)
            ndcg_results[f"nDCG@{k}"][method_name] = ndcg

    # --- RESULTS DISPLAY ---
    
    # Print detailed rankings table
    print("\nCompany Name           | Title                                                | Phase 1 Rank | Phase 2 Rank | LLM Rank")
    print("-" * 140)
    for rank2, (score2, posting) in enumerate(scored_postings, start=1):
        try:
            rank1_list = list(phase1_ids)
            original_index = ansel_postings.index(posting)
            rank1 = rank1_list.index(original_index) + 1
        except ValueError:
            rank1 = 'N/A'
        
        llm_rank = llm_ranks[original_index]
        
        rank1_str = f"{rank1:<12}" if isinstance(rank1, int) else f"{rank1:<12}"
        marker = ''
        if posting.company_name in offers_set:
            marker = '**'
        elif posting.company_name in interviews_set:
            marker = '*'
        
        print(f"{posting.company_name:<24} | {posting.title[:50]:<50} | {rank1_str} | {rank2:<12} | {llm_rank:<12} {marker}")

    # Print nDCG Results - Baselines
    print("\n" + "="*80)
    print("BASELINE METHODS - nDCG Analysis")
    print("="*80)
    print(f"{'Metric':<12} | {'Phase 1':<10} | {'Phase 2':<10} | {'LLM':<10}")
    print("-" * 80)
    
    for k in k_values:
        k_str = f"nDCG@{k}"
        print(f"{k_str:<12} | "
              f"{ndcg_results[k_str]['Phase 1']:<10.4f} | "
              f"{ndcg_results[k_str]['Phase 2']:<10.4f} | "
              f"{ndcg_results[k_str]['LLM']:<10.4f}")
    
    # Print nDCG Results - Best Ensembles
    print("\n" + "="*100)
    print("TOP 10 ENSEMBLE METHODS (sorted by nDCG@10)")
    print("="*100)
    
    # Find best methods at nDCG@10
    ensemble_only = {k: v for k, v in ndcg_results['nDCG@10'].items() 
                     if k not in ['Phase 1', 'Phase 2', 'LLM']}
    top_methods = sorted(ensemble_only.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"{'Method':<30} | {'nDCG@10':<10} | {'nDCG@20':<10} | {'nDCG@50':<10} | {'Lift@10':<10}")
    print("-" * 100)
    
    p1_baseline = ndcg_results['nDCG@10']['Phase 1']
    
    for method_name, _ in top_methods:
        ndcg_10 = ndcg_results['nDCG@10'][method_name]
        ndcg_20 = ndcg_results['nDCG@20'][method_name]
        ndcg_50 = ndcg_results['nDCG@50'][method_name]
        lift = ndcg_10 - p1_baseline
        
        print(f"{method_name:<30} | {ndcg_10:<10.4f} | {ndcg_20:<10.4f} | {ndcg_50:<10.4f} | {lift:<+10.4f}")
    
    print("="*100)
    
    # Summary statistics
    print("\nSUMMARY:")
    print(f"Best baseline method: Phase 1 (nDCG@10: {p1_baseline:.4f})")
    best_ensemble_name, best_ensemble_score = top_methods[0]
    print(f"Best ensemble method: {best_ensemble_name} (nDCG@10: {best_ensemble_score:.4f})")

    print(f"Improvement over baseline: {best_ensemble_score - p1_baseline:+.4f} ({(best_ensemble_score/p1_baseline - 1)*100:+.2f}%)")