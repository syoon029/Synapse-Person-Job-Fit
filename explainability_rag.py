import json
from typing import Dict, Any, Tuple

from database import get_postings_by_ids, get_resume_candidates, get_resume_text # DB helpers
from llm_starter_code import get_response
from llm_logging import _append_jsonl


def _safe_snip(text: str | None, n: int = 400) -> str:
    """Return a short preview of text; handle None safely."""
    if not text:
        return ""
    s = text.strip().replace("\n", " ")
    return (s[:n] + "...") if len(s) > n else s


def retrieve_candidates_from_resume(resume_id: int) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    """
    Load candidate info for a resume from DB.

    Expects database.get_resume_candidates(resume_id) to return:
    {
        "resume_id": 8,
        "candidate_list": [
            {"id": 12, "score": 0.91},
            {"id": 45, "score": 0.85},
            {"id": 72, "score": 0.82}
        ]
    }

    Then fetch all corresponding Posting rows from postings.db and
    build id -> Posting object mapping for downstream use.
    """
    entry = get_resume_candidates(resume_id)
    if not entry:
        raise ValueError(f"No candidate data found for resume_id={resume_id}")

    # Extract all job IDs
    all_ids = [c["id"] for c in entry["candidate_list"]]
    postings = get_postings_by_ids(all_ids)
    id2posting = {p.id: p for p in postings}

    return id2posting, entry



def build_rag_context(resume_text: str, id2posting: Dict[int, Any], entry: Dict[str, Any]) -> str:
    """
    Construct a compact, contrastive RAG context:
    - Resume text (provided by the caller)
    - Top-1 job (title/company/score/short description)
    - Other candidate jobs with scores (short description each)
    """
    candidates = entry["candidate_list"]
    if not candidates:
        raise ValueError("Candidate list is empty.")

    # Top-1 candidate
    top = candidates[0]
    top_post = id2posting.get(top["id"])
    if not top_post:
        raise RuntimeError(f"Posting {top['id']} not found in DB")

    ctx = []
    ctx.append(f"Resume:\n{_safe_snip(resume_text, 1200)}\n")
    ctx.append(
        f"Top-1 Job:\n"
        f"- Title: {top_post.title or ''}\n"
        f"- Company: {top_post.company_name or ''}\n"
        f"- Score: {top['score']:.3f}\n"
        f"- Description: {_safe_snip(top_post.description, 500)}\n\n"
    )

    # Add other candidates
    if len(candidates) > 1:
        ctx.append("Other Candidates:\n")
        for c in candidates[1:]:
            job = id2posting.get(c["id"])
            if not job:
                continue
            ctx.append(
                f"- Title: {job.title or ''}\n"
                f"  Company: {job.company_name or ''}\n"
                f"  Score: {c['score']:.3f}\n"
                f"  Description: {_safe_snip(job.description, 320)}\n"
            )

    context_text = "\n".join(ctx)
    return context_text



def generate_contrastive_explanation(context_text: str) -> str:
    """
    Ask the LLM to produce a contrastive explanation:
    Why is the top-ranked job a better match than the other candidates?
    The prompt encourages comparison, skill/resp alignment, and use of scores.
    """
    prompt = f"""
You are a job-matching analyst.
Using the following resume and multiple job candidates (with similarity scores),
explain in 4–6 sentences why the top-ranked job is the best match.

Context:
{context_text}

Instructions:
- Compare the top-1 job with the other candidates.
- Point out which skills, responsibilities, or domain focus align best with the resume.
- Reference similarity scores briefly as evidence (no need to restate all numbers).
- Be professional, specific, and concise. Avoid copying descriptions verbatim.
"""
    # Tune temperature/max_tokens if needed:
    return get_response(
        model_name="models/gemini-2.0-flash",
        prompt=prompt,
        max_tokens=400,
        temperature=0.3,
        logging=True,
        log_file="llm_history.jsonl",
    )

def explain_resume(resume_id: int) -> str:
    """
    End-to-end pipeline:
    1) Retrieve candidate jobs for a resume from DB
    2) Build RAG context (resume + jobs + scores)
    3) Generate contrastive explanation via LLM
    4) Log the result
    """
    resume_text = get_resume_text(resume_id)
    if not resume_text:
        raise ValueError(f"No resume text found for resume_id={resume_id}")

    id2posting, entry = retrieve_candidates_from_resume(resume_id)
    ctx = build_rag_context(resume_text, id2posting, entry)
    explanation = generate_contrastive_explanation(ctx)

    _append_jsonl("explanation_log.jsonl", {
        "resume_id": resume_id,
        "context": ctx,
        "explanation": explanation,
    })
    return explanation


if __name__ == "__main__":
    print(explain_resume(resume_id=8))
