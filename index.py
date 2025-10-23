import faiss
import numpy as np
from database import get_all_postings

def init_faiss_index() -> None:
    """
    FAISS index initialization.
    """
    postings = get_all_postings(embedded_only=True)
    ids = np.array([np.int64(posting["id"]) for posting in postings], dtype='int64')
    embeddings = np.array([np.array(posting["embedding"], dtype='float32') for posting in postings])
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