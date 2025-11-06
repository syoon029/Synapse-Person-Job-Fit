import os
import json
import numpy as np
from typing import Iterable, List, Optional
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_EMBED_MODEL)
    return _model

def _safe_join(parts: Iterable[str]) -> str:
    return " \n".join([p.strip() for p in parts if p and isinstance(p, str)])

def embed_text(text: str) -> np.ndarray:
    """
    Perform an SBERT vector on any text and return a 1D numpy array of float32.
    """
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=False)  
    return np.asarray(vec, dtype=np.float32)

def embed_posting(posting) -> str:
    """
    Embed the Posting object after field-level concatenation. Return a JSON string (float array) for writing to the Text field of the DB.
    """
    text = _safe_join([
        posting.title,
        posting.company_name,
        posting.skills_desc,
        posting.description,
        posting.formatted_experience_level,
        posting.location
    ])
    vec = embed_text(text)  # np.ndarray float32
    return json.dumps(vec.tolist()) 

def embed_resume_text(resume_text: str) -> np.ndarray:
    """
    Embed the full text of the resume
    """
    return embed_text(resume_text)
