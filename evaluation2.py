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
N_ITERS = 5


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


def evaluate_resume_once(resume_id: int, n_iters: int = N_ITERS) -> Optional[EvalRow]:
    """
On the same Phase-1 candidate set: original to multi-round lightweight optimization.
Each round continues mutate based on the resume of the previous round and prints the Phase-2 results of that round.
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

    id2rank_b = {pid: rank for rank, pid in enumerate(ids_b, start=1)}
    ranks_before = [id2rank_b[pid] for pid in candidate_ids]

    print(f"  - ID {resume_id}: BASE | Top1={top1_before_id} | Score={top1_before_score:.4f}")

    cur_text = base_text
    cur_top1_score = top1_before_score
    cur_ids = ids_b
    id2rank_cur = id2rank_b


    postings_ctx = get_postings_by_ids(candidate_ids[:3])

    changed_any = False  
    for it in range(1, n_iters + 1):
        cur_text = mutate_resume(cur_text, postings_ctx)

        scores_i, ids_i = _phase2_scores_for_text(cur_text, candidate_ids)
        top1_i_id, top1_i_score = ids_i[0], float(scores_i[0])

        id2rank_i = {pid: rank for rank, pid in enumerate(ids_i, start=1)}
        ranks_i = [id2rank_i[pid] for pid in candidate_ids]

        rho_vs0, _ = spearmanr(ranks_before, ranks_i)
        rho_vsp, _ = spearmanr([id2rank_cur[pid] for pid in candidate_ids], ranks_i)

        print(
            f"      Iter {it}: Top1={top1_i_id} | Score={top1_i_score:.4f} | "
            f"Δvs0={top1_i_score - top1_before_score:+.4f} | "
            f"ΔvsPrev={top1_i_score - cur_top1_score:+.4f} | "
            f"Top1ChangedFrom0={'YES' if top1_i_id != top1_before_id else 'no'} | "
            f"ρ(vs0)={float(rho_vs0) if rho_vs0 is not None else float('nan'):.3f} | "
            f"ρ(vsPrev)={float(rho_vsp) if rho_vsp is not None else float('nan'):.3f}"
        )

        if top1_i_id != top1_before_id:
            changed_any = True
        cur_top1_score = top1_i_score
        cur_ids = ids_i
        id2rank_cur = id2rank_i

    top1_after_id, top1_after_score = cur_ids[0], cur_top1_score
    rho_final, _ = spearmanr(ranks_before, [id2rank_cur[pid] for pid in candidate_ids])

    return EvalRow(
        resume_id=resume_id,
        k=len(candidate_ids),
        top1_before_id=top1_before_id,
        top1_before_score=top1_before_score,
        top1_after_id=top1_after_id,
        top1_after_score=float(top1_after_score),
        top1_changed=(top1_after_id != top1_before_id),
        top1_score_delta=float(top1_after_score - top1_before_score),
        spearman_rho=float(rho_final) if rho_final is not None else float("nan"),
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
