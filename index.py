import faiss
import numpy as np
from database import get_all_postings, embed_all_postings
import json

def extract_embedding(posting) -> np.ndarray:
    """
    Extract embedding from a posting.
    """
    return np.array(json.loads(posting.embedding), dtype='float32')

def compute_l2_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute L2 distance between two vectors.
    """
    return np.linalg.norm(vec1 - vec2)

def init_faiss_index(recompute_embeddings=True) -> None:
    """
    FAISS index initialization.
    """
    if recompute_embeddings:
        embed_all_postings()
        
    postings = get_all_postings(embedded_only=True)
    ids = np.array([np.int64(posting.id) for posting in postings], dtype='int64')

    # posting.embedding is stored as JSON string; convert to np.array float32
    embeddings = np.array([extract_embedding(posting) for posting in postings])
    faiss.normalize_L2(embeddings)
    embed_dim = embeddings.shape[1]

    index = faiss.IndexIDMap(faiss.IndexFlatIP(embed_dim))
    index.add_with_ids(embeddings, ids)

    with open('index.faiss', 'w'):
        faiss.write_index(index, 'index.faiss')

def search_faiss(search_vector, k=5) -> tuple[np.ndarray, np.ndarray]:
    """
    FAISS SEARCH
    """
    with open('index.faiss', 'r'):
        index = faiss.read_index('index.faiss')
    scores, ids = index.search(np.array(search_vector, dtype='float32').reshape(1, -1), k)

    return scores, ids