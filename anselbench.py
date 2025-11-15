import json
import numpy as np
import pandas as pd
import faiss
from database import Posting, Resume
from embed import embed_posting, embed_resume_text, _safe_join
from embed_stage2 import embed_text, doc_sim_score
from typing import List, Dict, Any
from sklearn.metrics import ndcg_score

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

    # --- ADDED: Prepare full Phase 1 score list for nDCG ---
    # We need a score for *all* N postings, not just the top 100.
    # We'll fill in the scores we know and use -inf for the rest.
    phase1_scores_all = np.full(len(ansel_postings), -np.inf, dtype=np.float32)
    for i in range(len(phase1_ids)):
        idx = phase1_ids[i]
        if idx >= 0:  # FAISS can return -1 if less than k results
            phase1_scores_all[idx] = phase1_scores[i]
    # Reshape for sklearn: (n_samples, n_items)
    phase1_scores_all_np = np.array([phase1_scores_all])
    # -------------------------------------------------------

    # PHASE 2 RECOMMENDATIONS
    print("Starting Phase 2 Recommendations...")
    scored_postings = []
    
    # --- ADDED: List to hold all P2 scores in original order ---
    phase2_scores_all = []
    # ----------------------------------------------------------
    
    phase2_resume_embedding = embed_text(ansel_resume.text)
    for i, posting in enumerate(ansel_postings):
        posting_embedding = embed_text(_safe_join([
            posting.title,
            posting.company_name,
            posting.skills_desc,
            posting.description,
            posting.formatted_experience_level,
            posting.location
        ]))
        
        score = doc_sim_score(phase2_resume_embedding, posting_embedding)
        scored_postings.append((score, posting))
        
        # --- ADDED: Store score in original list order ---
        phase2_scores_all.append(score)
        # -----------------------------------------------

    scored_postings.sort(reverse=True, key=lambda x: x[0])
    
    # --- ADDED: Reshape P2 scores for sklearn ---
    phase2_scores_all_np = np.array([phase2_scores_all])
    # --------------------------------------------

    # --- ADDED: nDCG Ground Truth and Calculation ---
    print("\nCalculating nDCG Lift...")
    
    # Create the ground truth relevance vector
    true_relevances = np.array([[
        get_relevance(p, offers_set, interviews_set) for p in ansel_postings
    ]])
    
    k_values = [10, 20, 50, 100]
    ndcg_results = {}
    
    for k in k_values:
        ndcg_p1 = ndcg_score(true_relevances, phase1_scores_all_np, k=k)
        ndcg_p2 = ndcg_score(true_relevances, phase2_scores_all_np, k=k)
        lift = ndcg_p2 - ndcg_p1
        
        ndcg_results[f"nDCG@{k}"] = {
            "Phase 1": ndcg_p1,
            "Phase 2": ndcg_p2,
            "Lift": lift
        }
    # ------------------------------------------------

    # ANALYTICS:
    # (Your existing analytics print block)
    print("Company Name           | Title                                                | Phase 1| Phase 2| Phase 1 Rank | Phase 2 Rank")
    for rank2, (score2, posting) in enumerate(scored_postings, start=1):
        try:
            rank1_list = list(phase1_ids)
            # Find index in the *original* postings list
            original_index = ansel_postings.index(posting)
            rank1 = rank1_list.index(original_index) + 1
            score1 = phase1_scores[rank1 - 1]
        except ValueError:
            rank1 = 'N/A'
            score1 = 'N/A'
        
        score1_str = f"{score1:.4f}" if isinstance(score1, float) else score1
        rank1_str = f"{rank1:<12}" if isinstance(rank1, int) else f"{rank1:<12}"
        
        print(f"{posting.company_name:<24} | {posting.title[:50]:<50} | {score1_str} | {score2:.4f} | {rank1_str} | {rank2} {'*' if posting.company_name in interviews_set else ''} {'**' if posting.company_name in offers_set else ''}")

    # (Your existing compute_stats and get_stats_for_companies functions)
    def compute_stats(ranks, scores):
        if not ranks:  # Handle empty lists
            return {
                'rank_mean': None, 'rank_median': None, 'rank_min': None, 'rank_max': None,
                'score_mean': None, 'score_median': None, 'score_min': None, 'score_max': None
            }
        ranks_array = np.array(ranks)
        scores_array = np.array(scores)
        return {
            'rank_mean': float(np.mean(ranks_array)),
            'rank_median': int(np.median(ranks_array)),
            'rank_min': int(np.min(ranks_array)),
            'rank_max': int(np.max(ranks_array)),
            'score_mean': float(np.mean(scores_array)),
            'score_median': float(np.median(scores_array)),
            'score_min': float(np.min(scores_array)),
            'score_max': float(np.max(scores_array)),
        }
    
    def get_stats_for_companies(company_list):
        phase1_ranks_new = []
        phase1_scores_new = []
        phase2_ranks_new = []
        phase2_scores_new = []
        for rank2, (score2, posting) in enumerate(scored_postings, start=1):
            if posting.company_name in company_list:
                try:
                    rank1_list = list(phase1_ids)
                    original_index = ansel_postings.index(posting)
                    rank1 = rank1_list.index(original_index) + 1
                    score1 = phase1_scores[rank1 - 1]
                except ValueError:
                    rank1 = None
                    score1 = None
                
                if rank1 is not None:
                    phase1_ranks_new.append(rank1)
                    phase1_scores_new.append(score1)
                phase2_ranks_new.append(rank2)
                phase2_scores_new.append(score2)
        return compute_stats(phase1_ranks_new, phase1_scores_new), compute_stats(phase2_ranks_new, phase2_scores_new)

    interview_stats_phase1, interview_stats_phase2 = get_stats_for_companies(set(interviews))
    offer_stats_phase1, offer_stats_phase2 = get_stats_for_companies(set(offers))

    print("\nInterviewed Companies Stats:")
    print("Phase 1:", interview_stats_phase1)
    print("Phase 2:", interview_stats_phase2)

    print("\nOffered Companies Stats:")
    print("Phase 1:", offer_stats_phase1)
    print("Phase 2:", offer_stats_phase2)

    # --- ADDED: Print nDCG Results Table ---
    print("\n" + "="*60)
    print("nDCG Lift Analysis (Relevance: Offer=2, Interview=1)")
    print(f"{'Metric':<10} | {'Phase 1 nDCG':<12} | {'Phase 2 nDCG':<12} | {'Lift':<12} | Status")
    print("-"*60)
    for k_str, scores in ndcg_results.items():
        status = '(IMPROVEMENT)' if scores['Lift'] > 0 else ('(REGRESSION)' if scores['Lift'] < 0 else '(NO CHANGE)')
        print(f"{k_str:<10} | {scores['Phase 1']:<12.4f} | {scores['Phase 2']:<12.4f} | {scores['Lift']:<+12.4f} | {status}")
    print("="*60)
    # ----------------------------------------