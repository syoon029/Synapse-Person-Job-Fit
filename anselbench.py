from database import Posting, Resume
from embed import embed_posting, embed_resume_text, _safe_join
from embed_stage2 import embed_text, doc_sim_score
import numpy as np
import pandas as pd
import faiss
import json

def read_ansel_resume(file_path='resume_data/ansel_resume.csv'):
    df = pd.read_csv(file_path)
    for row in df.itertuples():
        ansel_resume = Resume(id=1, text=row.Resume_str)
    return ansel_resume

def read_ansel_postings(file_path='linkedin_data/ansel_postings.csv'):
    df = pd.read_csv(file_path)
    postings = df.to_dict(orient='records')
    # print the first 50 characters of each field of each of the postings, each posting in one line, ignore newlines inside of posting description
    all_postings = []
    for posting in postings:
        posting_obj = Posting(
            title=posting['posting.title'],
            company_name=posting['posting.company_name'],
            skills_desc=posting['posting.skills_desc'],
            description=posting['posting.description'],
            formatted_experience_level=posting['posting.formatted_experience_level'],
            location=posting['posting.location']
        )
        all_postings.append(posting_obj)
    return all_postings

def initialize_faiss_index_for_ansel(postings):
    ids = np.array([i for i in range(len(postings))], dtype='int64')
    embeddings = np.array([np.array(json.loads(embed_posting(posting)), dtype='float32') for posting in postings])
    emb_dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(emb_dim))
    index.add_with_ids(embeddings, ids)
    return index

def companies_that_interviewed_ansel():
    interviews =  ["Baseten", "Google", "Palantir Technologies", "Databricks", "Headlands Tech", "Zip", "TikTok", "Applied Intuition", "Cape", "Optiver"]
    offers = ["Google", "Applied Intuition", "Baseten", "Palantir Technologies", "Databricks"]
    return interviews, offers

if __name__ == "__main__":
    print("Loading Data...")
    ansel_resume = read_ansel_resume()
    ansel_postings = read_ansel_postings()
    interviews, offers = companies_that_interviewed_ansel()

    print("Initializing FAISS Index...")
    faiss_index = initialize_faiss_index_for_ansel(ansel_postings)
    

    # PHASE 1 RECOMMENDATIONS
    print("Starting Phase 1 Recommendations...")
    phase1_resume_embedding = embed_resume_text(ansel_resume.text)
    scores, ids = faiss_index.search(phase1_resume_embedding.reshape(1, -1), 100)
    phase1_scores, phase1_ids = scores[0], ids[0]


    # PHASE 2 RECOMMENDATIONS
    print("Starting Phase 2 Recommendations...")
    scored_postings = []
    phase2_resume_embedding = embed_text(ansel_resume.text)
    for i, posting in enumerate(ansel_postings):
        posting_embedding = embed_text(_safe_join([
            posting.title,
            posting.company_name,
            posting.skills_desc,
            posting.description,
            posting.formatted_experience_level,
            posting.location
        ]))
        
        score = doc_sim_score(phase2_resume_embedding, posting_embedding)
        scored_postings.append((score, posting))

    scored_postings.sort(reverse=True, key=lambda x: x[0])

    # ANALYTICS:
    # FOR ALL resumes output Company Name | Phase 1 Score | Phase 2 Score | Phase 1 Rank | Phase 2 Rank
    print("Company Name             | Title                                              | Phase 1| Phase 2| Phase 1 Rank | Phase 2 Rank")
    for rank2, (score2, posting) in enumerate(scored_postings, start=1):
        try:
            rank1 = list(phase1_ids).index(ansel_postings.index(posting)) + 1
            score1 = phase1_scores[rank1 - 1]
        except ValueError:
            rank1 = 'N/A'
            score1 = 'N/A'
        # pad postings with spaces to align columns
        print(f"{posting.company_name:<24} | {posting.title[:50]:<50} | {score1:.4f} | {score2:.4f} | {rank1} | {rank2} {'*' if posting.company_name in interviews else ''} {'**' if posting.company_name in offers else ''}") 

    def compute_stats(ranks, scores):
        ranks_array = np.array(ranks)
        scores_array = np.array(scores)
        return {
            'rank_mean': float(np.mean(ranks_array)),
            'rank_median': int(np.median(ranks_array)),
            'rank_min': int(np.min(ranks_array)),
            'rank_max': int(np.max(ranks_array)),
            'score_mean': float(np.mean(scores_array)),
            'score_median': float(np.median(scores_array)),
            'score_min': float(np.min(scores_array)),
            'score_max': float(np.max(scores_array)),
        }
    
    def get_stats_for_companies(company_list):
        phase1_ranks_new = []
        phase1_scores_new = []
        phase2_ranks_new = []
        phase2_scores_new = []
        for rank2, (score2, posting) in enumerate(scored_postings, start=1):
            if posting.company_name in company_list:
                try:
                    rank1 = list(phase1_ids).index(ansel_postings.index(posting)) + 1
                    score1 = phase1_scores[rank1 - 1]
                except ValueError:
                    rank1 = None
                    score1 = None
                if rank1 is not None:
                    phase1_ranks_new.append(rank1)
                    phase1_scores_new.append(score1)
                phase2_ranks_new.append(rank2)
                phase2_scores_new.append(score2)
        return compute_stats(phase1_ranks_new, phase1_scores_new), compute_stats(phase2_ranks_new, phase2_scores_new)
    # Statistics: Mean, Median, Min, Max:  Phase 1 Rank, Phase 2 Rank, Phase 1 Score, Phase 2 Score for interviews vs. offer
    interview_stats_phase1, interview_stats_phase2 = get_stats_for_companies(set(interviews))
    offer_stats_phase1, offer_stats_phase2 = get_stats_for_companies(set(offers))

    print("\nInterviewed Companies Stats:")
    print("Phase 1:", interview_stats_phase1)
    print("Phase 2:", interview_stats_phase2)

    print("\nOffered Companies Stats:")
    print("Phase 1:", offer_stats_phase1)
    print("Phase 2:", offer_stats_phase2)