from embed import embed_resume_text
from index import search_faiss
from database import get_postings_by_ids, get_all_resumes
import random

"""
placeholder recommender module
"""

def recommend(resume):
    scores, ids = phase1_recommend(resume)

def phase1_recommend(resume):
    embedding = embed_resume_text(resume.text)
    scores, ids = search_faiss(embedding, k=10)
    return scores, ids

def phase2_recommend(resume, candidate_ids):
    postings = get_postings_by_ids(candidate_ids)
    scores = []
    ids = []
    return scores, ids


def test_phase1_recommend():
    resumes = get_all_resumes()

    # Randomly select five resumes using shuffle, seed 42 for reproducibility
    random.seed(42)
    random.shuffle(resumes)
    selected_resumes = resumes[:5]

    for resume in selected_resumes:
        print(f"Recommendations for Resume ID {resume.id}, Name: {resume.name}")
        # print resume text
        print(f"Resume Text: {resume.text[:100]}...")  # Print first 100 chars
        scores, ids = phase1_recommend(resume)
        ids = [int(id) for id in ids[0]]
        postings = get_postings_by_ids(ids)
        for score, posting in zip(scores[0], postings):
            print(f"  Score: {score:.4f}, Posting ID: {posting.id}, Title: {posting.title}")
        print()

if __name__ == "__main__":
    test_phase1_recommend()