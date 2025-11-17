import json
from typing import Dict, Any, Tuple, List

from database import get_postings_by_ids, get_resume_candidates, get_resume_text # DB helpers
from llm_starter_code import get_response
from llm_logging import _append_jsonl


def load_target_ids(path="resume_subset.json") -> List[int]:
    """Load resume IDs from JSON file."""
    try:
        with open(path, "r") as f:
            ids = json.load(f)
        print(f"[OK] Loaded {len(ids)} target resume IDs")
        return ids
    except Exception as e:
        print(f"[ERROR] Failed to load resume_subset.json: {e}")
        return []
    

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

    # sort and slice to top-3
    candidates = sorted(
        entry["candidate_list"],
        key=lambda x: x.get("phase2_advanced_score", -1.0),
        reverse=True
    )[:3]
    
    entry["candidate_list"] = candidates

    # fetch posting info
    all_ids = [c["id"] for c in candidates]
    postings = get_postings_by_ids(all_ids)
    id2posting = {p.id: p for p in postings}

    return id2posting, entry


def build_rag_context(resume_text: str, id2posting: Dict[int, Any], entry: Dict[str, Any]) -> str:
    """
    Construct a compact, contrastive RAG context:
    - Resume text (provided by the caller)
    - Top-1 job (title/company/score/short description)
    - Top-3 other candidate jobs with scores (short description each)
    """
    candidates = entry["candidate_list"]
    if not candidates:
        raise ValueError("Candidate list is empty.")

    # Top-1 candidate
    top = candidates[0]
    top_post = id2posting.get(top["id"])

    ctx = []
    ctx.append(f"Resume:\n{_safe_snip(resume_text, 1200)}\n")

    # Top-1
    ctx.append(
        f"Top-1 Job (Ensemble Score):\n"
        f"- Title: {top_post.title or ''}\n"
        f"- Company: {top_post.company_name or ''}\n"
        f"- Ensemble Score: {top.get('phase2_advanced_score', 0):.3f}\n"
        f"- Description: {_safe_snip(top_post.description, 500)}\n\n"
    )

    # Other top candidates
    if len(candidates) > 1:
        ctx.append("Other Top Candidates (Ensemble ranking):\n")
        for c in candidates[1:]:
            job = id2posting.get(c["id"])
            if not job:
                continue
            ctx.append(
                f"- Title: {job.title or ''}\n"
                f"  Company: {job.company_name or ''}\n"
                f"  Ensemble Score: {c.get('phase2_advanced_score', 0):.3f}\n"
                f"  Description: {_safe_snip(job.description, 320)}\n"
            )

    return "\n".join(ctx)


def build_rag_context_for_selected_postings(resume_text: str, postings: List[Any]) -> str:
    """
    Construct RAG context for UI-selected postings (no scores from DB).
    This version builds context from a list of Posting objects directly.
    
    Args:
        resume_text: The resume text
        postings: List of Posting objects selected by the user
    
    Returns:
        Formatted context string for LLM
    """
    if not postings:
        raise ValueError("No postings provided for explanation.")
    
    ctx = []
    ctx.append(f"Resume:\n{_safe_snip(resume_text, 1200)}\n")
    
    # Add all selected postings
    ctx.append(f"Selected Job Postings ({len(postings)} total):\n\n")
    
    for idx, posting in enumerate(postings, 1):
        ctx.append(
            f"Job {idx}:\n"
            f"- Title: {posting.title or ''}\n"
            f"- Company: {posting.company_name or ''}\n"
            f"- Location: {posting.location or 'Not specified'}\n"
            f"- Experience Level: {posting.formatted_experience_level or 'Not specified'}\n"
            f"- Skills: {_safe_snip(posting.skills_desc, 200)}\n"
            f"- Description: {_safe_snip(posting.description, 400)}\n\n"
        )
    
    return "\n".join(ctx)


def generate_contrastive_explanation(context_text: str) -> str:
    """
    Ask the LLM to produce a contrastive explanation:
    Why is the top-ranked job a better match than the other candidates?
    The prompt encourages comparison, skill/resp alignment, and use of scores.
    """
    prompt = f"""
You are a job-matching analyst.
Using the following resume and the top 3 job candidates (with similarity scores),
explain why the highest-scoring job (top-1) is the most suitable match,
while briefly comparing it with the other two candidates.

Context:
{context_text}

Instructions:
- Focus on explaining why the top-1 job fits best, based on both the resume and the scores.
- Discuss why the other jobs, while somewhat similar, are slightly less aligned.
- Emphasize skill, experience, and role alignment.
- Be analytical and concise (4–6 sentences).
"""

    # Tune temperature/max_tokens if needed:
    return get_response(
        model_name="models/gemini-2.0-flash",
        prompt=prompt,
        max_tokens=500,
        temperature=0.3,
        logging=True,
        log_file="llm_history.jsonl",
    )


def generate_multi_posting_explanation(context_text: str, num_postings: int) -> str:
    """
    Ask the LLM to explain why the selected postings are good matches for the resume.
    This is used for UI-selected postings where we explain each job's fit.
    
    Args:
        context_text: Formatted context with resume and postings
        num_postings: Number of postings being explained
    
    Returns:
        LLM-generated explanation
    """
    prompt = f"""
You are a job-matching analyst helping a candidate understand why certain jobs match their resume.

Context:
{context_text}

Instructions:
- Analyze how well each of the {num_postings} selected job posting(s) match the candidate's resume.
- For each job, explain:
  * What skills and experiences from the resume align with the job requirements
  * Why this role could be a good fit for the candidate
  * Any potential gaps or areas where the candidate might need to emphasize certain skills
- If multiple jobs are provided, briefly compare them and note which might be the strongest match.
- Be specific, citing actual skills, experiences, or qualifications from both the resume and job descriptions.
- Provide actionable insights the candidate can use.
- Keep the analysis concise but thorough (aim for 2-3 paragraphs per job).
"""

    return get_response(
        model_name="models/gemini-2.0-flash",
        prompt=prompt,
        max_tokens=1000,
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


def explain_recommendation(resume_text: str, postings: List[Any]) -> str:
    """
    UI-friendly function to explain why selected postings are good matches.
    This is called from the Streamlit UI with user-selected postings.
    
    Args:
        resume_text: The candidate's resume text
        postings: List of Posting objects selected by the user in the UI
    
    Returns:
        LLM-generated explanation of why these jobs match the resume
    """
    if not resume_text:
        raise ValueError("Resume text is required for explanation.")
    
    if not postings or len(postings) == 0:
        raise ValueError("At least one posting must be selected for explanation.")
    
    # Build context from selected postings
    ctx = build_rag_context_for_selected_postings(resume_text, postings)
    
    # Generate explanation
    explanation = generate_multi_posting_explanation(ctx, len(postings))
    
    # Log for analysis
    _append_jsonl("ui_explanation_log.jsonl", {
        "resume_preview": _safe_snip(resume_text, 200),
        "posting_ids": [p.id for p in postings],
        "num_postings": len(postings),
        "context": ctx,
        "explanation": explanation,
    })
    
    return explanation


if __name__ == "__main__":
    target_ids = load_target_ids()

    # Only first 5 for testing
    test_ids = target_ids[:5]
    print(f"Running explanation on resumes: {test_ids}")

    for rid in test_ids:
        print("\n====================================================")
        print(f"EXPLANATION FOR RESUME {rid}")
        print("====================================================")
        try:
            explanation = explain_resume(rid)
            print(explanation)
        except Exception as e:
            print(f"[ERROR] Failed for resume {rid}: {e}")