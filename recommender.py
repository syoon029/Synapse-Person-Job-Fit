from embed import embed_resume_text, _safe_join
from embed_stage2 import embed_text, doc_sim_score
from index import search_faiss
from database import get_postings_by_ids, get_all_resumes, save_resume_candidates
import random
import timeit

def pretty_print_recommendations(resume, scores, ids):
    print(f"Recommendations for Resume ID {resume.id}")
    print(f"Resume Text: {resume.text[:100]}...")  # Print first 100 chars
    postings = get_postings_by_ids(ids)
    for score, posting in zip(scores, postings):
        print(f"  Score: {score:.4f}, Posting ID: {posting.id}, Title: {posting.title}, Description: {posting.description[:50]}...")  # Print first 50 chars of description
    print()

def recommend(resume):
    scores, ids = phase1_recommend(resume)
    print("PHASE 1 _______________________________")
    pretty_print_recommendations(resume, scores[0], ids)
    scores2, ids2 = phase2_recommend(resume, ids)
    print("PHASE 2 _______________________________")
    pretty_print_recommendations(resume, scores2, ids2)

def phase1_recommend(resume):
    embedding = embed_resume_text(resume.text)
    scores, ids = search_faiss(embedding, k=10)
    save_resume_candidates(resume.id, ids[0].tolist(), scores[0].tolist()) # top k(10) resume list saved in DB
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
    selected_resumes = get_random_resumes()
    for resume in selected_resumes:
        recommend(resume)

if __name__ == "__main__":
    test_full_recommend()