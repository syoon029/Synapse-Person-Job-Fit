from embed import embed_resume_text, _safe_join
from embed_stage2 import embed_text, doc_sim_score
from index import search_faiss
from database import get_postings_by_ids, get_all_resumes, save_resume_candidates, update_phase2_scores,  update_phase2_advanced_scores, update_llm_scores
from llm_scorer import load_cache, pairwise_rank_to_scores
import random
import timeit
import numpy as np
from scipy.stats import rankdata
import time
from datetime import timedelta


# --- ADDED: Advanced Ensemble Functions ---

def normalize_scores_rank(scores_array):
    """
    Rank-based normalization (convert to percentile ranks)
    Returns 1D array normalized to [0, 1]
    """
    # Handle -inf by replacing with very low value
    scores_clean = scores_array.copy()
    scores_clean[scores_clean == -np.inf] = -999.0
    
    # Get ranks (higher score = higher rank)
    ranks = rankdata(scores_clean, method='average')
    
    # Normalize to [0, 1]
    if len(ranks) > 1:
        normalized = (ranks - 1) / (len(ranks) - 1)
    else:
        normalized = np.ones_like(ranks)
    
    return normalized

def ensemble_rank_weighted(phase1_scores, phase2_scores, llm_scores, weights=(0.8, 0.1, 0.1)):
    """
    Weighted average ensemble using rank-based normalization.
    This is the best performing method from anselbench.
    
    Args:
        phase1_scores: Array of Phase 1 scores
        phase2_scores: Array of Phase 2 scores  
        llm_scores: Array of LLM scores
        weights: Tuple of (w1, w2, w3) where sum should be 1.0
    
    Returns:
        Final ensemble scores (higher is better)
    """
    # Normalize each using rank-based method
    norm_p1 = normalize_scores_rank(phase1_scores)
    norm_p2 = normalize_scores_rank(phase2_scores)
    norm_llm = normalize_scores_rank(llm_scores)
    
    # Weighted average
    w1, w2, w3 = weights
    ensemble_scores = w1 * norm_p1 + w2 * norm_p2 + w3 * norm_llm
    
    return ensemble_scores

# --- MODIFIED: Core Recommendation Functions ---

def pretty_print_recommendations(resume, scores, ids):
    print(f"Recommendations for Resume ID {resume.id}")
    print(f"Resume Text: {resume.text[:100]}...")
    postings = get_postings_by_ids(ids)
    for score, posting in zip(scores, postings):
        print(f"  Score: {score:.4f}, Posting ID: {posting.id}, Title: {posting.title}, Description: {posting.description[:50]}...")
    print()

def recommend(resume):
    scores, ids = phase1_recommend(resume)
    print("PHASE 1 _______________________________")
    pretty_print_recommendations(resume, scores[0], ids)
    scores2, ids2 = phase2_recommend(resume, ids)
    print("PHASE 2 _______________________________")
    update_phase2_scores(resume.id, list(ids2), list(scores2))
    pretty_print_recommendations(resume, scores2, ids2)


def recommend_advanced(resume):
    """
    Advanced recommendation using ensemble of Phase 1, Phase 2, and LLM.
    Uses the best performing ensemble: WAvg_rank_80-10-10
    """
    scores1, ids1 = phase1_recommend(resume)
    print("PHASE 1 _______________________________")
    pretty_print_recommendations(resume, scores1[0], ids1)
    
    scores_advanced, ids_advanced = phase2_recommend_advanced(resume, ids1)
    print("PHASE 2 ADVANCED (Ensemble) ___________")
    update_phase2_scores(resume.id, list(ids_advanced), list(scores_advanced))
    pretty_print_recommendations(resume, scores_advanced, ids_advanced)



def phase1_recommend(resume):
    embedding = embed_resume_text(resume.text)
    scores, ids = search_faiss(embedding, k=10)
    save_resume_candidates(resume.id, ids[0].tolist(), scores[0].tolist())
    return scores, [int(id) for id in ids[0]]


def phase2_recommend(resume, candidate_ids):
    postings = get_postings_by_ids(candidate_ids)
    scored_postings = []
    resume_embedding = embed_text(resume.text)
    for posting in postings:
        posting_embedding = embed_text(_safe_join([
            posting.title,
            posting.company_name,
            posting.skills_desc,
            posting.description,
            posting.formatted_experience_level,
            posting.location
        ]))
        
        score = doc_sim_score(resume_embedding, posting_embedding)
        scored_postings.append((score, posting.id))
    scored_postings.sort(reverse=True, key=lambda x: x[0])
    scores, ids = zip(*scored_postings)
    return scores, ids


def phase2_recommend_advanced(resume, candidate_ids):
    """
    Advanced Phase 2 that combines Phase 1, Phase 2, and LLM rankings
    using the best ensemble method: Rank-based normalization with 80-10-10 weights.
    
    Args:
        resume: Resume object
        candidate_ids: List of posting IDs from Phase 1
    
    Returns:
        (scores, ids): Tuple of final ensemble scores and posting IDs (sorted by score)
    """
    print(f"\n[Phase 2 Advanced] Processing {len(candidate_ids)} candidates...")
    
    # Get postings
    postings = get_postings_by_ids(candidate_ids)
    
    # Create ID to index mapping
    id_to_idx = {pid: idx for idx, pid in enumerate(candidate_ids)}
    
    # --- 1. Get Phase 1 scores (already computed) ---
    # Since these came from FAISS search, we need to preserve the original scores
    # We'll use the order as a proxy for scores (first = best)
    phase1_scores = np.array([len(candidate_ids) - i for i in range(len(candidate_ids))], dtype=float)
    
    # --- 2. Compute Phase 2 scores ---
    print("[Phase 2 Advanced] Computing Phase 2 embeddings...")
    phase2_scores = np.zeros(len(candidate_ids))
    resume_embedding = embed_text(resume.text)
    
    for posting in postings:
        posting_embedding = embed_text(_safe_join([
            posting.title,
            posting.company_name,
            posting.skills_desc,
            posting.description,
            posting.formatted_experience_level,
            posting.location
        ]))
        
        score = doc_sim_score(resume_embedding, posting_embedding)
        idx = id_to_idx[posting.id]
        phase2_scores[idx] = score
    
    # --- 3. Compute LLM scores using pairwise ranking ---
    print("[Phase 2 Advanced] Computing LLM pairwise rankings...")
    
    # Load cache
    llm_cache = load_cache()
    
    # Prepare posting texts
    postings_texts = []
    for posting in postings:
        posting_text = _safe_join([
            f"Title: {posting.title}",
            f"Company: {posting.company_name}",
            f"Location: {posting.location}",
            f"Experience Level: {posting.formatted_experience_level}",
            f"Skills: {posting.skills_desc}",
            f"Description: {posting.description}"
        ])
        postings_texts.append(posting_text)
    
    # Get LLM scores (this may take time for cache misses)
    llm_scores = np.array(pairwise_rank_to_scores(
        resume_text=resume.text,
        postings_texts=postings_texts,
        cache=llm_cache
    ))
    
    # --- 4. Ensemble with rank-based normalization (80-10-10) ---
    print("[Phase 2 Advanced] Computing ensemble scores...")
    ensemble_scores = ensemble_rank_weighted(
        phase1_scores=phase1_scores,
        phase2_scores=phase2_scores,
        llm_scores=llm_scores,
        weights=(0.8, 0.1, 0.1)  # Best performing weights from anselbench
    )
    
    # --- 5. Sort by ensemble scores ---
    scored_postings = list(zip(ensemble_scores, candidate_ids))
    scored_postings.sort(reverse=True, key=lambda x: x[0])
    
    final_scores, final_ids = zip(*scored_postings)
    
    print(f"[Phase 2 Advanced] Complete. Top posting ID: {final_ids[0]} with score: {final_scores[0]:.4f}")
    
    return final_scores, final_ids

# --- Utility Functions ---

def get_random_resumes(n=5):
    resumes = get_all_resumes()
    random.seed(42)
    random.shuffle(resumes)
    return resumes[:n]

def test_phase1_recommend():
    selected_resumes = get_random_resumes()
    for resume in selected_resumes:
        scores, ids = phase1_recommend(resume)
        pretty_print_recommendations(resume, scores[0], ids)

    average_time = timeit.timeit(
        stmt='phase1_recommend(resume)',
        setup='from __main__ import phase1_recommend; resume=selected_resumes[0]',
        number=10,
        globals={'selected_resumes': selected_resumes}
    ) / 10

    print(f"Average time for phase1_recommend: {average_time:.4f} seconds")

def test_full_recommend():
    selected_resumes = get_random_resumes(5)
    total = len(selected_resumes)
    
    print(f"\n[INFO] Starting full recommendation for {total} resumes...\n")
    
    start_time = time.time()

    for idx, resume in enumerate(selected_resumes, start=1):

        print("="*80)
        print(f"[ {idx} / {total} ] Processing Resume ID {resume.id}")


        elapsed = time.time() - start_time
        avg_time_per_resume = elapsed / idx
        remaining = avg_time_per_resume * (total - idx)
        
        eta_str = str(timedelta(seconds=int(remaining)))
        print(f"[ETA] Estimated time remaining: {eta_str}")
        print("="*80)

        
        recommend(resume)
        
    total_elapsed = time.time() - start_time
    print("\n[INFO] All resumes processed!")
    print(f"Total time: {str(timedelta(seconds=int(total_elapsed)))}")

def test_full_recommend_advanced(n=1):
    """
    Test the advanced ensemble recommendation system
    with real-time progress + ETA.
    """
    selected_resumes = get_all_resumes()
    total = len(selected_resumes)

    print(f"\n[INFO] Starting ADVANCED recommendation for {total} resumes...\n")

    start_time = time.time()

    for idx, resume in enumerate(selected_resumes, start=1):
        print("\n" + "=" * 80)
        print(f"[ {idx} / {total} ] Testing Advanced Recommendation for Resume {resume.id}")

        elapsed = time.time() - start_time
        avg_time_per_resume = elapsed / idx
        remaining = avg_time_per_resume * (total - idx)
        eta_str = str(timedelta(seconds=int(remaining)))

        print(f"[ETA] Estimated time remaining: {eta_str}")
        print("=" * 80)
        
        recommend_advanced(resume)

    total_elapsed = time.time() - start_time
    print("\n[INFO] All advanced recommendations complete!")
    print(f"Total time: {str(timedelta(seconds=int(total_elapsed)))}")



def recommend_full_pipeline(resume):
    print(f"\n==============================")
    print(f"Running FULL PIPELINE for Resume {resume.id}")
    print(f"==============================")

    # --------------------
    # 1) Phase 1
    # --------------------
    p1_scores_raw, p1_ids_raw = phase1_recommend(resume)
    p1_scores = p1_scores_raw[0].tolist()
    p1_ids = p1_ids_raw

    print("[Phase 1] Done.")

    # --------------------
    # 2) Phase 2  (only once)
    # --------------------
    phase2_scores_raw, _ = phase2_recommend(resume, p1_ids)
    phase2_scores = list(phase2_scores_raw)

    update_phase2_scores(resume.id, p1_ids, phase2_scores)
    print("[Phase 2] Done.")

    # --------------------
    # 3) LLM (only once)
    # --------------------
    postings = get_postings_by_ids(p1_ids)
    postings_texts = []
    for p in postings:
        postings_texts.append(_safe_join([
            f"Title: {p.title}",
            f"Company: {p.company_name}",
            f"Location: {p.location}",
            f"Experience Level: {p.formatted_experience_level}",
            f"Skills: {p.skills_desc}",
            f"Description: {p.description}"
        ]))

    llm_cache = load_cache()
    llm_scores = np.array(pairwise_rank_to_scores(
        resume_text=resume.text,
        postings_texts=postings_texts,
        cache=llm_cache
    ))
    
    llm_scores = [float(s) for s in llm_scores]

    update_llm_scores(resume.id, p1_ids, llm_scores)
    print("[LLM] Done.")

    # --------------------
    # 4) Advanced Ensemble (NO recomputation)
    # --------------------
    ensemble_scores = ensemble_rank_weighted(
        phase1_scores=np.array(p1_scores),
        phase2_scores=np.array(phase2_scores),
        llm_scores=np.array(llm_scores),
        weights=(0.8, 0.1, 0.1)
    )

    scored_postings = list(zip(ensemble_scores, p1_ids))
    scored_postings.sort(reverse=True, key=lambda x: x[0])
    adv_scores, adv_ids = zip(*scored_postings)

    adv_scores = [float(s) for s in adv_scores]
    adv_ids = list(adv_ids)

    update_phase2_advanced_scores(resume.id, adv_ids, adv_scores)
    print("[Advanced Ensemble] Done.")

    return {
        "phase1_ids": p1_ids,
        "phase1_scores": p1_scores,
        "phase2_scores": phase2_scores,
        "llm_scores": llm_scores,
        "advanced_ids": adv_ids,
        "advanced_scores": adv_scores
    }



def run_full_pipeline_all():
    resumes = get_all_resumes()
    total = len(resumes)
    start = time.time()

    for idx, r in enumerate(resumes, 1):
        elapsed = time.time() - start
        avg = elapsed / idx
        remaining = avg * (total - idx)
        eta_str = str(timedelta(seconds=int(remaining)))

        print("\n===================================================")
        print(f"[{idx}/{total}] Resume ID={r.id}  | ETA: {eta_str}")
        print("===================================================\n")

        recommend_full_pipeline(r)

    total_elapsed = time.time() - start
    print("\nAll Done! Total time:", str(timedelta(seconds=int(total_elapsed))))



if __name__ == "__main__":
    # Uncomment the test you want to run:
    #test_full_recommend()  # Original system
    #test_full_recommend_advanced()  # New advanced system
    run_full_pipeline_all() # do everything 