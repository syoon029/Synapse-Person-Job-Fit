# phase2_iter_eval.py

from __future__ import annotations
import json
import time
from datetime import timedelta
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
from scipy.stats import spearmanr

from database import (
    get_resume_text,
    get_resume_candidates,
    get_postings_by_ids,
)
from embed import _safe_join
from embed_stage2 import embed_text, doc_sim_score
from resume_optimization import mutate_resume


TOPK_FROM_PHASE1 = 2
N_ITERS = 5
SUBSET_JSON = "resume_subset.json"


def load_subset_ids() -> List[int]:
    with open(SUBSET_JSON, "r") as f:
        return json.load(f)


def _phase2_scores_for_text(resume_text: str, candidate_ids: List[int]):
    postings = get_postings_by_ids(candidate_ids)
    resume_emb = embed_text(resume_text)

    scored = []
    for p in postings:
        p_emb = embed_text(_safe_join([
            p.title, p.company_name, p.skills_desc,
            p.description, p.formatted_experience_level, p.location
        ]))
        s = doc_sim_score(resume_emb, p_emb)
        scored.append((s, p.id))

    scored.sort(key=lambda x: x[0], reverse=True)
    scores, ids = zip(*scored)
    return np.array(scores), list(ids)


def _take_topk_from_db(resume_id: int) -> Optional[List[int]]:
    entry = get_resume_candidates(resume_id)
    if not entry or not entry.get("candidate_list"):
        return None
    ids = [c["id"] for c in entry["candidate_list"]][:TOPK_FROM_PHASE1]
    return ids


@dataclass
class ResumeIterRecord:
    top1_scores: List[float]
    spearman_rhos: List[float]


def evaluate_resume_iters(resume_id: int) -> Optional[ResumeIterRecord]:
    base_text = get_resume_text(resume_id)
    if not base_text:
        return None

    candidate_ids = _take_topk_from_db(resume_id)
    if not candidate_ids:
        return None

    # === Iter 0: base resume ===
    scores0, ids0 = _phase2_scores_for_text(base_text, candidate_ids)
    top1_before = float(scores0[0])        # Iter 0 score
    id2rank_before = {pid: r for r, pid in enumerate(ids0, start=1)}
    ranks_before = [id2rank_before[pid] for pid in candidate_ids]

    top1_scores = [top1_before]     # iteration 0 
    spearman_rhos = [1.0]           # self-correlation = 1

    current_text = base_text
    current_ranks = ranks_before

    postings_ctx = get_postings_by_ids(candidate_ids[:3])

    # === Iter 1~N ===
    for _ in range(N_ITERS):
        current_text = mutate_resume(current_text, postings_ctx)

        scores_i, ids_i = _phase2_scores_for_text(current_text, candidate_ids)
        top1_scores.append(float(scores_i[0]))

        id2rank_i = {pid: r for r, pid in enumerate(ids_i, start=1)}
        ranks_i = [id2rank_i[pid] for pid in candidate_ids]

        rho, _ = spearmanr(ranks_before, ranks_i)
        spearman_rhos.append(float(rho) if rho is not None else np.nan)

        current_ranks = ranks_i

    return ResumeIterRecord(
        top1_scores=top1_scores,
        spearman_rhos=spearman_rhos
    )


# ==================================================================================
# FINAL EVALUATION: AVERAGE OVER ALL RESUMES
# ==================================================================================

def evaluate_all_resumes_iteration_curve():
    subset_ids = load_subset_ids()
    subset_ids = subset_ids[:10]
    print(f"Running Phase-2 Iteration Evaluation for {len(subset_ids)} resumes...\n")
    

    start_time = time.time()

    all_records: List[ResumeIterRecord] = []

    for rid in subset_ids:
        rec = evaluate_resume_iters(rid)
        if rec:
            all_records.append(rec)

    total_evaluated = len(all_records)
    print(f"Total valid resumes evaluated: {total_evaluated}")

    if total_evaluated == 0:
        print("No valid data.")
        return

    # ============================
    # Compute iteration-wise averages
    # ============================
    iter_avg_score = []
    iter_avg_delta = []
    iter_avg_rho = []

    for it in range(N_ITERS + 1):
        scores = [rec.top1_scores[it] for rec in all_records]
        rhos = [rec.spearman_rhos[it] for rec in all_records]

        avg_score = float(np.mean(scores))
        avg_delta = avg_score - float(np.mean([rec.top1_scores[0] for rec in all_records]))
        avg_rho = float(np.nanmean(rhos))

        iter_avg_score.append(avg_score)
        iter_avg_delta.append(avg_delta)
        iter_avg_rho.append(avg_rho)

    # ============================
    # PRINT RESULTS
    # ============================
    elapsed = timedelta(seconds=int(time.time() - start_time))

    print("\n========================================================")
    print("Phase-2 Optimization Trend (Averaged over all resumes)")
    print(f"Total Time: {elapsed}")
    print("========================================================")
    print("Iter | Avg Top1 Score | Avg Δvs0 | Avg Spearman ρ")
    print("--------------------------------------------------------")

    for it in range(N_ITERS + 1):
        print(f"{it:>4} | {iter_avg_score[it]:.4f}        | "
              f"{iter_avg_delta[it]:+.4f}   | {iter_avg_rho[it]:.3f}")

    # peak iteration
    peak_iter = int(np.argmax(iter_avg_delta))
    peak_gain = iter_avg_delta[peak_iter]

    print("--------------------------------------------------------")
    print(f"Peak improvement at iteration {peak_iter}")
    print(f"Max average Δ = {peak_gain:+.4f}")
    print("========================================================\n")


if __name__ == "__main__":
    evaluate_all_resumes_iteration_curve()
