from typing import Union, List

import numpy as np
import torch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel

from database import Posting, Resume

_EMBED_MODEL = 'roberta-base'

def _clean_text(text: str) -> str:
    '''
    Remove special characters, whitespace, etc. from the raw text.
    
    Currently the following operations are performed:
    - Remove newlines ('\n')
    - Remove extra whitespace and replace with single space
    '''
    
    return ' '.join(text.replace('\n', '').split())
    

def cosine_similarity(arr1: np.ndarray, arr2: np.ndarray) -> float:
    '''arr1 and arr2 should be 1D vectors'''
    return (np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2)))
    
    
def fetch_all_postings_text(database_url: str) -> List[str]:
    '''
    Returns a list of the postings' body text from the database
    '''
    engine = create_engine(database_url)
    with Session(engine) as session:
        stmt = select(Posting.description)
        return list(session.scalars(stmt))
    
    
def fetch_all_resumes_text(database_url: str) -> List[str]:
    '''
    Returns a list of the resumes' body text from the database
    '''
    engine = create_engine(database_url)
    with Session(engine) as session:
        stmt = select(Resume.text)
        return list(session.scalars(stmt))
    
    
def embed_text(text: str, word_reduction_type: str='sum_last_four') -> np.ndarray:
    '''
    Get array of word embeddings for a single document (text). Word embeddings can be computed from BERT's layer outputs
    in several ways (word_reduction_type):
    - sum_last_four: Sum the outputs of the last four layers
    - concat_last_four: Concatenate the outputs of the last four layers in one long vector (NOT YET IMPLEMENTED)
    - last_layer: Use only the lasy layer as the embedding (NOT YET IMPLEMENTED)
    
    Returns embeddings as torch.Tensor: Size (num_tokens x 768); one vector for each word.
    '''
    text = _clean_text(text)
    tokenizer = AutoTokenizer.from_pretrained(_EMBED_MODEL)
    # Tokenize with truncation to the model's max length
    tokenized = tokenizer(text, truncation=True, max_length=512, return_tensors='pt')

    model = AutoModel.from_pretrained(_EMBED_MODEL, add_pooling_layer=False, output_hidden_states=True)
    model.eval()

    with torch.no_grad():
        outputs = model(**tokenized)
        # outputs.hidden_states is a tuple: one tensor per layer (batch, seq_len, hidden)
        hidden_states = outputs.hidden_states

        if word_reduction_type == 'sum_last_four':
            # Take the last 4 layers, stack and sum them: result shape (batch, seq_len, hidden)
            last_four = torch.stack(hidden_states[-4:], dim=0)
            summed = last_four.sum(dim=0)
            # batch size is 1, return token-level embeddings for that single example
            embeddings = summed[0]
            return embeddings.numpy()
        else:
            # Fallback: return last layer token embeddings
            embeddings = hidden_states[-1][0]
            return embeddings.numpy()

    
def doc_sim_score(emb1: np.ndarray, emb2: np.ndarray, comp_type: str='mean_embeddings') -> np.ndarray:
    '''
    Get the similarity score between a job posting and a resume (order does not matter). Parameters
    emb1 and emb2 are token-level embeddings as returned by `embed_text`. There are several
    options to compare (comp_type):
    
    - mean_embeddings: Average token embeddings for each document, then take the cosine similarity
    - others????
    '''
    
    if comp_type == 'mean_embeddings':
        mean1, mean2 = emb1.mean(axis=0), emb2.mean(axis=0)
        return cosine_similarity(mean1, mean2)
    else:
        raise ValueError('Please provide a valid value for argument `mean_embeddings`')