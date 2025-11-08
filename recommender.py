from embed import embed_resume_text
from index import search_faiss
from database import get_postings_by_ids, get_all_resumes
import random
import timeit

"""
placeholder recommender module
"""

def recommend(resume):
    scores, ids = phase1_recommend(resume)

def phase1_recommend(resume):
    embedding = embed_resume_text(resume.text)
    scores, ids = search_faiss(embedding, k=10)
    return scores, [int(id) for id in ids[0]]

def phase2_recommend(resume, candidate_ids):
    postings = get_postings_by_ids(candidate_ids)
    scores = []
    ids = []
    return scores, ids

def get_random_resumes(n=5):
    resumes = get_all_resumes()
    random.seed(42)
    random.shuffle(resumes)
    return resumes[:n]

def test_phase1_recommend():
    selected_resumes = get_random_resumes()
    for resume in selected_resumes:
        print(f"Recommendations for Resume ID {resume.id}, Name: {resume.name}")
        # print resume text
        print(f"Resume Text: {resume.text[:100]}...")  # Print first 100 chars
        scores, ids = phase1_recommend(resume)
        postings = get_postings_by_ids(ids)
        for score, posting in zip(scores[0], postings):
            print(f"  Score: {score:.4f}, Posting ID: {posting.id}, Title: {posting.title}")
        print()

    average_time = timeit.timeit(
        stmt='phase1_recommend(resume)',
        setup='from __main__ import phase1_recommend; resume=resumes[0]',
        number=10,
        globals={'resumes': resumes}
    ) / 10

    print(f"Average time for phase1_recommend: {average_time:.4f} seconds")

if __name__ == "__main__":
    test_phase1_recommend()