import json
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.stats import rankdata
import time
from datetime import timedelta
import random

# --- External Library: Sentence Transformer for Title Embeddings ---
from sentence_transformers import SentenceTransformer
# --- Database functions ---
from database import get_all_resumes, get_resume_candidates, get_posting_by_id, Posting, Resume 

# --- GLOBAL MODEL SETUP ---
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TITLE_EMBEDDER = None
try:
    TITLE_EMBEDDER = SentenceTransformer(_EMBED_MODEL)
    print(f"Loaded title embedding model: {_EMBED_MODEL}")
except Exception as e:
    print(f"FATAL ERROR: Failed to load SentenceTransformer. Error: {e}")


# ------------------------------------------------------------
# NEW: Load target resume IDs from resume_subset.json
# ------------------------------------------------------------
def load_target_ids(path="resume_subset.json"):
    """
    Loads the list of resume IDs from resume_subset.json.
    Used for target-only evaluations.
    """
    try:
        with open(path, "r") as f:
            ids = json.load(f)
        print(f"[OK] Loaded {len(ids)} target resume IDs")
        return ids
    except Exception as e:
        print(f"[ERROR] Failed to load resume_subset.json: {e}")
        return []


# --------------------------------------------------------------------------------
# A. HELPER FUNCTIONS
# --------------------------------------------------------------------------------

def get_random_resumes(n=5):
    resumes = get_all_resumes()
    random.seed(42)
    random.shuffle(resumes)
    return resumes[:n]


def get_max_candidate(candidate_list: List[Dict[str, Any]], score_key: str) -> Dict[str, Any]:
    return max(candidate_list, key=lambda x: x.get(score_key, -1.0))


def _get_resume_title_from_text(resume_text: str) -> str | None:
    """
    Extract main resume title from the first line of resume text.
    """
    if not resume_text:
        return None
    first_line = resume_text.strip().split('\n')[0].strip()
    if not first_line:
        return None

    stop_phrases = [
        'SUMMARY','PROFESSIONAL PROFILE','CAREER OVERVIEW','EXPERIENCE','SKILLS',
        'PROFILE','OBJECTIVE','ACCOMPLISHMENTS','HIGHLIGHTS','QUALIFICATIONS',
        'EDUCATION','EXECUTIVE','OVERVIEW','WORK HISTORY','ASSOCIATE',
        'PROFESSIONAL','MANAGER','DIRECTOR','SPECIALIST'
    ]
    
    title = first_line
    title_upper = title.upper()
    min_stop = len(title)

    # Early stop removal
    for phrase in stop_phrases:
        idx = title_upper.find(phrase)
        if idx != -1 and (idx + len(phrase) == len(title_upper) or title_upper[idx+len(phrase)].isspace()):
            min_stop = min(min_stop, idx)

    if min_stop < len(title):
        title = title[:min_stop].strip()

    # Remove trailing noise
    title_upper = title.upper()
    for phrase in stop_phrases:
        if title_upper.endswith(phrase):
            title = title[:title_upper.rfind(phrase)].strip()
            title_upper = title.upper()

    title = re.sub(r'[\s\-/,.:;]+$', '', title).strip()

    if len(title) > 60:
        return title[:60].rsplit(' ',1)[0]

    if not title:
        parts = first_line.split()
        return parts[0] if parts else None

    return title


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b))


# --------------------------------------------------------------------------------
# B. Ranking Metrics (Title Similarity Ground Truth)
# --------------------------------------------------------------------------------

def calculate_ranking_metrics_by_title_sim(candidate_list, resume_title, k_values):
    if TITLE_EMBEDDER is None or not resume_title:
        return {}

    score_phases = {
        "Phase1": "phase1_score",
        "Phase2": "phase2_score",
        "Ensemble": "phase2_advanced_score"
    }
    
    results = {}

    resume_emb = TITLE_EMBEDDER.encode(resume_title, normalize_embeddings=True)

    # Compute relevance scores
    candidate_relevance = []
    for cand in candidate_list:
        posting = get_posting_by_id(cand.get("id"))
        rel = 0.0
        if posting and posting.title:
            emb = TITLE_EMBEDDER.encode(posting.title, normalize_embeddings=True)
            rel = cosine_similarity(resume_emb, emb)
        candidate_relevance.append(rel)

    # Evaluate each scoring phase
    for phase_name, score_key in score_phases.items():

        scores_for_sort = [(cand.get(score_key, -1.0), idx) for idx, cand in enumerate(candidate_list)]
        sorted_idx = [idx for score, idx in sorted(scores_for_sort, key=lambda x: x[0], reverse=True)]
        relevance_sorted = [candidate_relevance[i] for i in sorted_idx]

        for K in k_values:
            if K > len(relevance_sorted):
                continue

            rel_k = relevance_sorted[:K]

            # NDCG@K
            dcg = np.sum(rel_k / np.log2(np.arange(2, K+2)))
            ideal = sorted(candidate_relevance, reverse=True)[:K]
            idcg = np.sum(np.array(ideal) / np.log2(np.arange(2, K+2)))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            # Binary metrics
            thresh = 0.7
            binary_k = [1 if r >= thresh else 0 for r in rel_k]
            total_binary = sum(1 if r >= thresh else 0 for r in candidate_relevance)

            P = sum(binary_k) / K
            R = sum(binary_k) / total_binary if total_binary > 0 else 0.0
            F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

            if phase_name not in results:
                results[phase_name] = {}

            results[phase_name][f"P@{K}"] = P
            results[phase_name][f"R@{K}"] = R
            results[phase_name][f"F1@{K}"] = F1
            results[phase_name][f"NDCG@{K}"] = ndcg

    return results


# --------------------------------------------------------------------------------
# C. Random evaluation (unchanged)
# --------------------------------------------------------------------------------
def evaluate_ranking_metrics_by_title(k_values=[1,5,10]):
    if TITLE_EMBEDDER is None:
        print("Model load failed.")
        return {}
    start = time.time()
    
    resumes = get_random_resumes(5)
    cumulative = {}
    total_eval = 0

    print(f"Starting Title Similarity Ranking Evaluation for {len(resumes)} random resumes.\n")

    for resume in resumes:
        title = _get_resume_title_from_text(resume.text)
        if not title:
            continue

        data = get_resume_candidates(resume.id)
        if not data:
            continue

        metrics = calculate_ranking_metrics_by_title_sim(data["candidate_list"], title, k_values)
        if metrics and "Phase1" in metrics:
            total_eval += 1
            for phase, vals in metrics.items():
                if phase not in cumulative:
                    cumulative[phase] = {k:0.0 for k in vals.keys()}
                for k,v in vals.items():
                    cumulative[phase][k] += v

    elapsed = timedelta(seconds=int(time.time()-start))

    print("\n" + "="*80)
    print(f"Final Ranking Metrics (Averaged over {total_eval} resumes)")
    print(f"Total Evaluation Time: {elapsed}")
    print("="*80)

    if total_eval == 0:
        print("No data.")
        return {}

    final_avg = {phase: {k:v/total_eval for k,v in vals.items()} for phase, vals in cumulative.items()}

    print(f"{'Metric':<10} | {'Phase 1':<10} | {'Phase 2':<10} | {'Ensemble':<10}")
    print("-"*45)

    metrics_sorted = sorted(cumulative["Phase1"].keys())
    for key in metrics_sorted:
        print(f"{key:<10} | {final_avg['Phase1'][key]:.4f}     | {final_avg['Phase2'][key]:.4f}     | {final_avg['Ensemble'][key]:.4f}")

    print("="*80)
    return final_avg


# --------------------------------------------------------------------------------
# D. TARGET evaluation (MODIFIED to use resume_subset.json)
# --------------------------------------------------------------------------------
def evaluate_ranking_metrics_by_title_target(k_values=[1,5,10]):
    target_ids = load_target_ids()

    if TITLE_EMBEDDER is None:
        print("Model load failed.")
        return {}

    start = time.time()
    resumes = [r for r in get_all_resumes() if r.id in target_ids]

    cumulative = {}
    total_eval = 0

    print(f"Starting Title Similarity Ranking Evaluation for {len(resumes)} target resumes.")
    print(f"Target IDs: {target_ids}\n")

    for resume in resumes:
        title = _get_resume_title_from_text(resume.text)
        if not title:
            continue

        data = get_resume_candidates(resume.id)
        if not data:
            continue

        metrics = calculate_ranking_metrics_by_title_sim(data["candidate_list"], title, k_values)

        if metrics and "Phase1" in metrics:
            total_eval += 1
            for phase, vals in metrics.items():
                if phase not in cumulative:
                    cumulative[phase] = {k:0.0 for k in vals.keys()}
                for k,v in vals.items():
                    cumulative[phase][k] += v

    elapsed = timedelta(seconds=int(time.time()-start))

    print("\n" + "="*80)
    print(f"Final Ranking Metrics (Averaged over {total_eval} resumes)")
    print(f"Total Evaluation Time: {elapsed}")
    print("="*80)

    if total_eval == 0:
        print("No data.")
        return {}

    final_avg = {phase: {k:v/total_eval for k,v in vals.items()} for phase, vals in cumulative.items()}

    print(f"{'Metric':<10} | {'Phase 1':<10} | {'Phase 2':<10} | {'Ensemble':<10}")
    print("-"*45)

    metrics_sorted = sorted(cumulative["Phase1"].keys())
    for key in metrics_sorted:
        print(f"{key:<10} | {final_avg['Phase1'][key]:.4f}     | {final_avg['Phase2'][key]:.4f}     | {final_avg['Ensemble'][key]:.4f}")

    print("="*80)
    return final_avg


# --------------------------------------------------------------------------------
# E. Debug for one resume (unchanged)
# --------------------------------------------------------------------------------

def debug_sorted_indices_target(resume_id: int):
    data = get_resume_candidates(resume_id)
    cand = data["candidate_list"]

    print(f"\n==== Debug sorted indices for Resume {resume_id} ====\n")
    for key in ["phase1_score","phase2_score","phase2_advanced_score"]:
        idx_sorted = sorted(range(len(cand)), key=lambda i: cand[i][key], reverse=True)
        print(f"[{key}] indices:", idx_sorted)
        print(f"[{key}] scores:", [round(cand[i][key],4) for i in idx_sorted])
        print()


# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    # Evaluate ONLY resumes listed in resume_subset.json
    evaluate_ranking_metrics_by_title_target()

    # Optional: debug one
    # debug_sorted_indices_target(10062724)
