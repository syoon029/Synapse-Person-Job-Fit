from embed import embed_resume
from index import search_faiss
from database import get_postings_by_ids
"""
placeholder recommender module
"""

def recommend(resume):
    scores, ids = phase1_recommend(resume)

def phase1_recommend(resume):
    embedding = embed_resume(resume)
    scores, ids = search_faiss(embedding, k=20)
    return scores, ids

def phase2_recommend(resume, candidate_ids):
    postings = get_postings_by_ids(candidate_ids)
    scores = []
    ids = []
    return scores, ids