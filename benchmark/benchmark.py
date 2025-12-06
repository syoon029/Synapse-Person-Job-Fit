import os
import numpy as np
import pandas as pd
import time
import argparse
import warnings
warnings.simplefilter("ignore", UserWarning)

from tqdm import tqdm

LIMIT = 999999

from infrastructure.database import (
    get_all_postings, 
    get_all_resumes,
    get_postings_by_ids,
    import_postings_from_csv,
    embed_all_postings,
    init_db,
    update_db,
    delete_all_postings,
    Resume
)

from infrastructure.index import init_faiss_index, search_faiss, set_faiss_path
from embeddings.embed import embed_resume_text
from recommender import phase2_recommend, recommend_full_pipeline
from explainability_rag import explain_recommendation

POSTINGS_DB_PATH = 'postings.db'
RESUMES_DB_PATH = 'resumes.db'


class Timer:
    # Convenience class to time a function
    def __init__(self):
        self.start_t = None
    def restart(self):
        self.start_t = time.perf_counter()
    def stop(self, restart=False):
        elapsed = time.perf_counter() - self.start_t
        if restart:
            self.restart()
        return elapsed

    
def append_to_npz(fname, key, arr):
    # Append an array to a NPZ file
    if not os.path.exists(fname):
        np.savez(fname, **{key: arr})
        return
    
    data = dict(np.load(fname))
    data[key] = arr
    
    np.savez(fname, **data)
    
    
def change_db(db_path):
    '''Change the current DATABASE_URL to the given .db file'''
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
    update_db()

    
def benchmark_init_posting_embed(num_runs):
    '''
    The initial embedding of all available job postings. Benchmark includes the import into the database
    as well as the encodings via the SBERT model.
    '''
    global POSTINGS_DB_PATH
    POSTINGS_DB_PATH = 'benchmark/' + POSTINGS_DB_PATH
    change_db(POSTINGS_DB_PATH)
    times = []
    timer = Timer()
    
    print(f'Importing into {POSTINGS_DB_PATH}')
    
    for i in range(num_runs):
        print(f'Run #{i+1}: ', end='')
        import_postings_from_csv('/home/hice1/khom9/scratch/CS6220_Project/postings.csv')
#         import_postings_from_csv('/home/hice1/khom9/CS6220/Synapse-Person-Job-Fit/temp/postings5k.csv')
    
        timer.restart()
        embed_all_postings()
        
        times.append(timer.stop())
        print(f'{times[-1]:.5f}s')
        
        delete_all_postings()
        
    os.remove(POSTINGS_DB_PATH)    
    return times


def benchmark_init_faiss(num_runs):
    '''
    Bench mark the initialization of the FAISS index database with the pre-computed embeddings.
    
    The posting database must be already initialized with embeddings.
    '''
    times = []
    timer = Timer()
    change_db(POSTINGS_DB_PATH)
    print(f'Reading resumes from {POSTINGS_DB_PATH}')
    
    for i in range(num_runs):
        print(f'Run #{i+1}: ', end='')
        
        timer.restart()
        init_faiss_index(recompute_embeddings=False)
        times.append(timer.stop())
        print(f'{times[-1]:.5f}s')
                
    return times


def benchmark_embed_stage1(num_runs):
    '''
    Benchmark the time it takes to compute the embedding of a single resume. 
    `num_runs` specifies the number of passes over the entire resume database.
    
    The resume database must be already initialized.
    '''
    times = []
    timer = Timer()
    change_db(RESUMES_DB_PATH)
    print(f'Reading resumes from {RESUMES_DB_PATH}')
    resumes = get_all_resumes()
    for r in range(num_runs):
        for i, resume in tqdm(list(enumerate(resumes[:LIMIT]))):
            timer.restart()
            embed_resume_text(resume.text)
            times.append(timer.stop())
    
    return times
    
    
def benchmark_sim_search(num_runs):
    '''
    Benchmark the time it takes to search for the top 20 nearest postings given a resume with Phase 1. 
    `num_runs` specifies the number of passes over the entire resume database.
    '''
    times = []
    timer = Timer()
    change_db(RESUMES_DB_PATH)
    print(f'Reading resumes from {RESUMES_DB_PATH}')
    resumes = get_all_resumes()
    
    for r in range(num_runs):
        print(f'Run #{r+1}')
        for resume in tqdm(resumes[:LIMIT]):
            emb = embed_resume_text(resume.text)
            timer.restart()
            search_faiss(emb, k=20)
            times.append(timer.stop())
    
    return times


def benchmark_stage2(num_runs):
    '''
    Benchmark the time it takes to compute the Phase 2 similarity between a resume and 20 postings
    retrieved from Phase 1.
    `num_runs` specifies the number of passes over the entire resume database.
    '''
    times = []
    timer = Timer()
    print(f'Reading postings from {POSTINGS_DB_PATH}')
    change_db(RESUMES_DB_PATH)
    resumes = get_all_resumes()
    change_db(POSTINGS_DB_PATH)
    
    for r in range(num_runs):
        print(f'Run #{r+1}')
        for resume in tqdm(resumes[:LIMIT//20]):
            emb = embed_resume_text(resume.text)
            scores, ids = search_faiss(emb, k=20)
            
            for posting_id in list(map(int, ids.flatten().tolist())):
                timer.restart()
                phase2_recommend(resume, [posting_id])
                times.append(timer.stop())

    return times


def benchmark_explain_rag(num_runs):
    '''
    Benchmark the time taken to run the LLM explainability with RAG.
    `num_runs` specifies the number of passes over the limited sample of resumes
    '''
    times = []
    timer = Timer()
    print(f'Reading postings from {POSTINGS_DB_PATH}')
    change_db(RESUMES_DB_PATH)
    resumes = get_all_resumes()
    change_db(POSTINGS_DB_PATH)
    limit = 25 # 100 calls to the API to reduce costs
    
    for r in range(num_runs):
        print(f'Run #{r+1}')
        for i,resume in enumerate(tqdm(resumes[:limit])):
            emb = embed_resume_text(resume.text)
            scores, ids = search_faiss(emb, k=20)
            postings = get_postings_by_ids(list(map(int, ids.flatten().tolist())))

            timer.restart()
            explain_recommendation(resume.text, postings)
            times.append(timer.stop())
            print(f'Run #{r+1} ({i+1}) Took {times[-1]}s')
    
    return times


def benchmark_full_pipeline(num_runs):
    '''
    Benchmark the time for a full run of the pipeline
    '''
    times = []
    timer = Timer()
    init_db()
    print(f'Reading postings from {POSTINGS_DB_PATH}')
    change_db(RESUMES_DB_PATH)
    resumes = get_all_resumes()
    change_db(POSTINGS_DB_PATH)
    limit = 25 # 100 calls to the API to reduce costs
    
    for i,resume in enumerate(tqdm(resumes[:limit])):
        timer.restart()
        recommend_full_pipeline(resume)
        times.append(timer.stop())
        print(f'Resume {i+1}: Took {times[-1]}s')
        
    return times
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark various portions of the software. Meant to be run from the home directory `Synapse-Person-Job-Fit`')
    parser.add_argument('--section', type=str, help='Section to benchmark')
    parser.add_argument('--num', type=int, default=1, help='Number of repetitions to run')
    parser.add_argument('--save-file', type=str, help='NumPy npz file to add the time results to')
    parser.add_argument('--db-prefix', type=str, default='', help='Prefix to add to the names of the .db files created by this benchmark during `init_embed`')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--limit', type=int, default=999999)
    
    args = parser.parse_args()    
    
    set_faiss_path(args.db_prefix + 'index.faiss')
    if args.section == 'init_embed':
        POSTINGS_DB_PATH = args.db_prefix + POSTINGS_DB_PATH
    if not args.verbose:
        tqdm = lambda x: x
    if args.limit:
        LIMIT = args.limit
    
    funcs = {
        'init_embed': benchmark_init_posting_embed,
        'init_faiss': benchmark_init_faiss,
        'embed_stage1': benchmark_embed_stage1,
        'sim_search': benchmark_sim_search,
        'stage2': benchmark_stage2,
        'explain_rag': benchmark_explain_rag,
        'full': benchmark_full_pipeline
    }
    
    if args.section not in funcs:
        raise ValueError(f'Invalid section name! Valid options are {list(funcs.keys())}')
    
    print(f'Running section "{args.section}" {args.num} times')
    times = funcs[args.section](args.num)
    if args.save_file is not None:
        append_to_npz(args.save_file, args.section, np.array(times))
        print(f'Saved as "{args.section}" to {args.save_file}')
    else:
        print(times)
