import json
from typing import List, Dict, Any
from database import get_resume_candidates

def load_resume_subset(path="resume_subset.json"):
    try:
        with open(path, "r") as f:
            ids = json.load(f)
        print(f"[OK] Loaded {len(ids)} resume IDs from {path}")
        return ids
    except Exception as e:
        print(f"[ERROR] Could not load resume_subset.json: {e}")
        return []

def get_top_candidate(candidates: List[Dict[str, Any]], field: str):
    return max(candidates, key=lambda x: x.get(field, -1.0))

def check_reranking(resume_id: int):
    """
    Returns whether each stage (P2, LLM, Ensemble) improved
    over Phase 1 using that stage's scoring metric.
    """
    data = get_resume_candidates(resume_id)
    if not data:
        return None

    cands = data["candidate_list"]

    # Get top-1 picks for each score type
    p1 = get_top_candidate(cands, "phase1_score")
    p2 = get_top_candidate(cands, "phase2_score")
    llm = get_top_candidate(cands, "llm_score")
    ens = get_top_candidate(cands, "phase2_advanced_score")

    # Compute improvements relative to Phase 1
    imp_p2 = p2["phase2_score"] - p1.get("phase2_score", -1)
    imp_llm = llm["llm_score"] - p1.get("llm_score", -1)
    imp_ens = ens["phase2_advanced_score"] - p1.get("phase2_advanced_score", -1)

    return {
        "p2": imp_p2 > 0,
        "llm": imp_llm > 0,
        "ens": imp_ens > 0
    }


# ---------------------------------------------------------
# Summary evaluation for only the subset of resume IDs
# ---------------------------------------------------------
def evaluate_subset_summary():
    """
    Loads resume_subset.json and evaluates P2 / LLM / Ensemble
    reranking success across only those resumes.
    """
    target_ids = load_resume_subset()
    print(f"\nEvaluating {len(target_ids)} resumes...\n")

    total = 0
    p2_count = 0
    llm_count = 0
    ens_count = 0

    for rid in target_ids:
        result = check_reranking(rid)
        if not result:
            continue

        total += 1
        if result["p2"]:
            p2_count += 1
        if result["llm"]:
            llm_count += 1
        if result["ens"]:
            ens_count += 1

    # ----- Final Summary -----
    print("\n========================================")
    print("       Comprehensive Reranking Summary")
    print("========================================")
    print(f"Total evaluated: {total}\n")

    if total == 0:
        print("No resumes evaluated.")
        print("========================================\n")
        return

    print("Phase 2 Reranking:")
    print(f"  Success count: {p2_count} / {total}")
    print(f"  Success rate : {p2_count / total * 100:.2f}%\n")

    print("LLM Reranking:")
    print(f"  Success count: {llm_count} / {total}")
    print(f"  Success rate : {llm_count / total * 100:.2f}%\n")

    print("Ensemble (0.8/0.1/0.1):")
    print(f"  Success count: {ens_count} / {total}")
    print(f"  Success rate : {ens_count / total * 100:.2f}%")
    print("========================================\n")


# ---------------------------------------------------------
# Main Entry
# ---------------------------------------------------------
if __name__ == "__main__":
    evaluate_subset_summary()
