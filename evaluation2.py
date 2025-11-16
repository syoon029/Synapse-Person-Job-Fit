from __future__ import annotations
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from database import (
    get_all_resumes,
    get_resume_text,
    get_resume_candidates,
    get_postings_by_ids,
)
from embed import _safe_join
from embed_stage2 import embed_text, doc_sim_score
from resume_optimization import mutate_resume

TOPK_FROM_PHASE1 = 2  


def _phase2_scores_for_text(resume_text: str, candidate_ids: List[int]) -> Tuple[np.ndarray, List[int]]:
    """
    Return: In descending order of Phase-2 scores (scores, ids)
    """
    postings = get_postings_by_ids(candidate_ids)
    resume_emb = embed_text(resume_text)

    scored = []
    for p in postings:
        p_emb = embed_text(_safe_join([
            p.title,
            p.company_name,
            p.skills_desc,
            p.description,
            p.formatted_experience_level,
            p.location
        ]))
        s = doc_sim_score(resume_emb, p_emb)
        scored.append((s, p.id))

    scored.sort(key=lambda x: x[0], reverse=True)
    scores, ids = zip(*scored)
    return np.array(scores, dtype=float), list(ids)


def _take_topk_from_db(resume_id: int) -> Optional[List[int]]:
    """
    Read the candidates of this resume in the DB (the same as evaluation.py) and take the first K ids.
If there is no data, return None (the caller skips it by themselves).
    """
    entry = get_resume_candidates(resume_id)
    if not entry or not entry.get("candidate_list"):
        return None
    ids = [c["id"] for c in entry["candidate_list"]][:TOPK_FROM_PHASE1]
    return ids


@dataclass
class EvalRow:
    resume_id: int
    k: int
    top1_before_id: int
    top1_before_score: float
    top1_after_id: int
    top1_after_score: float
    top1_changed: bool
    top1_score_delta: float
    spearman_rho: float


def evaluate_resume_once(resume_id: int) -> Optional[EvalRow]:
    """
    On the same set of Phase-1 candidates: original vs. lightweight optimized Phase-2 comparison.
    """
    base_text = get_resume_text(resume_id)
    if not base_text or not base_text.strip():
        print(f"  - ID {resume_id}: No resume text. Skip.")
        return None

    candidate_ids = _take_topk_from_db(resume_id)
    if not candidate_ids:
        print(f"  - ID {resume_id}: No candidate data found. Run phase1 first. Skip.")
        return None

    scores_b, ids_b = _phase2_scores_for_text(base_text, candidate_ids)
    top1_before_id, top1_before_score = ids_b[0], float(scores_b[0])

    postings_ctx = get_postings_by_ids(candidate_ids[:3]) 
    opt_text = mutate_resume(base_text, postings_ctx)

    scores_a, ids_a = _phase2_scores_for_text(opt_text, candidate_ids)
    top1_after_id, top1_after_score = ids_a[0], float(scores_a[0])

    id2rank_b = {pid: rank for rank, pid in enumerate(ids_b, start=1)}
    id2rank_a = {pid: rank for rank, pid in enumerate(ids_a, start=1)}
    ranks_before = [id2rank_b[pid] for pid in candidate_ids]
    ranks_after  = [id2rank_a[pid] for pid in candidate_ids]
    rho, _ = spearmanr(ranks_before, ranks_after)

    return EvalRow(
        resume_id=resume_id,
        k=len(candidate_ids),
        top1_before_id=top1_before_id,
        top1_before_score=top1_before_score,
        top1_after_id=top1_after_id,
        top1_after_score=top1_after_score,
        top1_changed=(top1_before_id != top1_after_id),
        top1_score_delta=(top1_after_score - top1_before_score),
        spearman_rho=float(rho) if rho is not None else float("nan"),
    )


def evaluate_all_resumes_optimization():
    all_resumes = get_all_resumes()
    print(f"Starting Phase-2 vs Optimized evaluation for {len(all_resumes)} resumes.")

    results: List[EvalRow] = []
    changed = 0

    for r in all_resumes:
        try:
            rec = evaluate_resume_once(r.id)
            if rec is None:
                continue
            results.append(rec)
            status = "CHANGED" if rec.top1_changed else "STABLE"
            if rec.top1_changed:
                changed += 1
            print(f"  - ID {rec.resume_id}: {status} | ΔTop1={rec.top1_score_delta:+.4f} | ρ={rec.spearman_rho:.3f}")
        except Exception as e:
            print(f"  - ID {r.id}: ERROR - {e}")

    print("\n" + "=" * 50)
    print("Phase-2 Robustness to Resume Optimization — Summary")
    print(f"Total Resumes Evaluated: {len(results)}")

    if results:
        pct_changed = 100.0 * changed / len(results)
        mean_delta = float(np.mean([x.top1_score_delta for x in results]))
        mean_rho = float(np.mean([x.spearman_rho for x in results]))
        print(f"Top-1 changed: {changed} ({pct_changed:.2f}%)")
        print(f"Mean ΔTop1 score: {mean_delta:+.4f}")
        print(f"Mean Spearman ρ: {mean_rho:.3f}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    evaluate_all_resumes_optimization()
