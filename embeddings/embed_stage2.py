import time
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from multiprocessing import Pool, Manager
from transformers import RobertaTokenizer, RobertaModel, BatchEncoding
from datasets import Dataset as HFDataset, DatasetDict, load_dataset, concatenate_datasets
from datasets.dataset_dict import DatasetDict

from infrastructure.database import Posting, Resume
from embeddings.contrastive_learning import fetch_cl_model, soft_alignment_similarity

# Name of tokenizer (will be downloaded from Huggingface)
_TOKENIZER_NAME = 'roberta-base'

# Path to the trained CL model
_MODEL_PATH = './contrastive_learning.pth'

BLOCK_SIZE = 128 # Stride length when splitting long texts into 512-length segments
MAX_SEQ_LEN = 510 # maximum length of a sequence that BERT can operate on (510 tokens + the start and end tokens)
POSTING_TXT_COL = 'description' # Name of the column in the postings CSV that contains the raw text
RESUME_TXT_COL = 'Resume_str' # Name of the column in the resume CSV that contains the raw text

DEVICE = ('cuda:0' if torch.cuda.is_available() else 'cpu')
if DEVICE == 'cpu':
    print('WARNING! Running PyTorch models on CPU may result in long inference times!')

TOKENIZER = RobertaTokenizer.from_pretrained(_TOKENIZER_NAME)
# BERT_MODEL =  RobertaModel.from_pretrained(_BERT_MODEL_PATH, add_pooling_layer=False, output_hidden_states=True)

MODEL = fetch_cl_model(_MODEL_PATH, device=DEVICE)



def _clean_text(text: str) -> str:
    '''
    Remove special characters, whitespace, etc. from the raw text.
    
    Currently the following operations are performed:
    - Remove newlines ('\n')
    - Remove extra whitespace and replace with single space
    '''
    
    return ' '.join(text.replace('\n', '').split())

    
def fetch_all_postings_text(database_url: str) -> list[str]:
    '''
    Returns a list of the postings' body text from the database
    '''
    engine = create_engine(database_url)
    with Session(engine) as session:
        stmt = select(Posting.description)
        return list(session.scalars(stmt))
    
    
def fetch_all_resumes_text(database_url: str) -> list[str]:
    '''
    Returns a list of the resumes' body text from the database
    '''
    engine = create_engine(database_url)
    with Session(engine) as session:
        stmt = select(Resume.text)
        return list(session.scalars(stmt))
    
    
def doc_sim_score(txt1, txt2, dist_func='soft_align'):
    '''
    Get the document similarity score with a model trained with contrastive learning. Depending on the similarity metric chosen, a higher value
    can imply higher or lower similarity.
    
    txt1: String representing the first text to compare.
    txt2: String representing the second text to compare.
    device: The device to use for computations (str).
    dist_func: The name of the distance function to use. Choices are:
            - 'soft_token_align': Soft token alignment. Computes distance at the token level.
            - 'mean_emb_dist': L2 distance between the mean token embedding vectors for each document. 
    '''    
    tokenized1 = TOKENIZER(txt1, truncation=True, padding='max_length', max_length=MAX_SEQ_LEN+2, return_tensors='pt')
    tokenized2 = TOKENIZER(txt2, truncation=True, padding='max_length', max_length=MAX_SEQ_LEN+2, return_tensors='pt')

    MODEL.eval()
    with torch.no_grad():
        emb1 = MODEL(tokenized1.to(DEVICE))
        emb2 = MODEL(tokenized2.to(DEVICE))
    
    if dist_func == 'soft_align':
        emb1 = F.normalize(emb1, dim=2)
        emb2 = F.normalize(emb2, dim=2)
        return 0.5 * (
            soft_alignment_similarity(emb1, emb2, tokenized1['attention_mask'], tokenized2['attention_mask']).cpu().item() + 
            soft_alignment_similarity(emb2, emb1, tokenized2['attention_mask'], tokenized1['attention_mask']).cpu().item()
        )
    elif dist_func == 'mean_emb_dist':
        emb1 = embedding_mean(emb1, tokenized1['attention_mask'])
        emb2 = embedding_mean(emb2, tokenized2['attention_mask'])

        return torch.norm(emb1 - emb2).cpu().item()
    else:
        raise ValueError()

    
def embed_text_OLD(text: str, word_reduction_type: str='sum_last_four') -> np.ndarray:
    '''
    DEPRECATED. UNUSED. NOT UP-TO-DATE.
    
    Get array of word embeddings for a single document (text). Word embeddings can be computed from BERT's layer outputs
    in several ways (word_reduction_type):
    - sum_last_four: Sum the outputs of the last four layers
    - concat_last_four: Concatenate the outputs of the last four layers in one long vector (NOT YET IMPLEMENTED)
    - last_layer: Use only the lasy layer as the embedding (NOT YET IMPLEMENTED)
    
    Returns embeddings as torch.Tensor: Size (num_tokens x 768); one vector for each word.
    '''
    text = _clean_text(text)
#     tokenizer = AutoTokenizer.from_pretrained(_EMBED_TOKENIZER)
    # Tokenize with truncation to the model's max length
    tokenized = TOKENIZER(text, truncation=True, max_length=512, return_tensors='pt')

    model = AutoModel.from_pretrained(_EMBED_MODEL, add_pooling_layer=False, output_hidden_states=True)
    model.eval()

    with torch.no_grad():
        outputs = model(**tokenized)
        # outputs.hidden_states is a tuple: one tensor per layer (batch, seq_len, hidden)
        hidden_states = outputs.hidden_states

        if word_reduction_type == 'sum_last_four':
            # Take the last 4 layers, stack and sum them: result shape (batch, seq_len, hidden)
            last_four = torch.stack(hidden_states[-4:], dim=0).sum(dim=0)
            embeddings = last_four[0][1:-1]
        elif word_reduction_type == 'concat_last_four':
            # Take the last four layers and concatenate them into one long vector
            embeddings = torch.cat(hidden_states[-4:], dim=2).squeeze()[1:-1]
        else:
            # Fallback: return last layer token embeddings
            embeddings = hidden_states[-1][0]
        
        return embeddings.numpy()

    
def doc_sim_score_OLD(emb1: np.ndarray, emb2: np.ndarray, comp_type: str='mean_embeddings_cosine') -> np.ndarray:
    '''
    DEPRECATED. UNUSED.
    
    Get the similarity score between a job posting and a resume (order does not matter). Parameters
    emb1 and emb2 are token-level embeddings as returned by `embed_text`. There are several
    options to compare (comp_type). 
    
    IMPORTANT: Depending on the `comp_type`, a higher value can either imply higher or lower similarity!.
    
    - mean_embeddings_cosine: Average token embeddings for each document, then take the cosine similarity. 
                              Higher values imply higher similarity.
    - pairwise_cosine: Average of maximum cosine sim for each token in one document as compared to all tokens
                       in the other document. Higher values imply higher similarity.
    - wmd: Word Mover's Distance. Lower values imply higher similarity.
    
    return: The similarity score between the two document embeddings.
    '''
    
    if comp_type == 'mean_embeddings_cosine':
        mean1, mean2 = emb1.mean(axis=0), emb2.mean(axis=0)
        return cosine_similarity(mean1, mean2)
    elif comp_type == 'pairwise_cosine':
        s1, s2 = 0, 0
        for i in range(emb1.shape[0]):
            s1 += max(cosine_similarity(emb1[i], emb2[j]) for j in range(emb2.shape[0]))
        for j in range(emb2.shape[0]):
            s2 += max(cosine_similarity(emb1[i], emb2[j]) for i in range(emb1.shape[0]))
        return 0.5 * (s1 / emb1.shape[0] + s2 / emb2.shape[0])
    elif comp_type == 'wmd':
        return word_movers_distance(emb1, emb2)
    else:
        raise ValueError('Please provide a valid value for argument `comp_type`')

        
        
