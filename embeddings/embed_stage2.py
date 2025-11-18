from typing import Union, List

import time
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from multiprocessing import Pool, Manager
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel, BatchEncoding
from datasets import Dataset as HFDataset, DatasetDict, load_dataset, concatenate_datasets
from datasets.dataset_dict import DatasetDict

from infrastructure.database import Posting, Resume

_EMBED_TOKENIZER = 'roberta-base'
_EMBED_MODEL = './roberta-tuned'

BLOCK_SIZE = 128 # Stride length when splitting long texts into 512-length segments
MAX_SEQ_LEN = 510 # maximum length of a sequence that BERT can operate on (510 tokens + the start and end tokens)
POSTING_TXT_COL = 'description'
RESUME_TXT_COL = 'Resume_str'

tokenizer = RobertaTokenizer.from_pretrained(_EMBED_TOKENIZER)

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


def word_movers_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    # Compute pairwise distance matrix between token embeddings
    M = ot.dist(emb1, emb2, metric='euclidean')
    M /= M.max()  # Normalize for stability

    # Uniform weights (bag-of-words)
    a = np.ones(emb1.shape[0]) / emb1.shape[0]
    b = np.ones(emb2.shape[0]) / emb2.shape[0]

    # Compute Earth Mover's Distance (Wasserstein-1)
    distance = ot.emd2(a, b, M)
    return distance


def preprocess_posting(x):
    '''preprocess for a job listing, as read from the linkedin CSV file'''
    return tokenizer(x[POSTING_TXT_COL], add_special_tokens=False)


def preprocess_resume(x):
    '''preprocess for a resume CSV file'''
    return tokenizer(x[RESUME_TXT_COL], add_special_tokens=False)


def group_texts(token_data, return_tensors=None):
    '''
    Split texts that are two long (longer than 512 tokens) into overlapping segments with a sliding window of stride 128,
    and window size of 512. For example, a text with 1376 tokens is broken down into 8 segments that start at the 
    following indices:
        [0, 128, 256, 384, 512, 640, 768, 896]
    Then each segment has size:
        [512, 512, 512, 512, 512, 512, 512, 480]
        
    Input is standard dict of keys {'input_ids', 'attention_mask'}
    '''
    result = {k: [] for k in token_data.keys()}
    all_keys, all_data = zip(*list(token_data.items()))
#     print(all_data)
    for data_tuple in zip(*all_data):
        # Data tuple consists of one list for each key in the data (should just be 'input_ids' and 'attention_mask' of equal length)         
        # Not a general assumption that this is what is inside data_tuple
#         print(data_tuple)
        token_list, attn_mask = data_tuple
        n_tokens = len(token_list)
        shifts = [t for t in range(0, n_tokens, BLOCK_SIZE) if t+MAX_SEQ_LEN+2-n_tokens < BLOCK_SIZE]
        if len(shifts) == 0:
            shifts = [0]
        for shift in shifts:
            # Re-encode the token list with the special eos/bos characters
            enc_token_list = tokenizer(
                tokenizer.decode(token_list[shift : shift+MAX_SEQ_LEN]), 
                max_length=MAX_SEQ_LEN+2,
                padding='max_length',
                padding_side='right',
                return_tensors=return_tensors,
                return_token_type_ids=False,
                return_overflowing_tokens=False, 
                return_special_tokens_mask=False,
                return_length=False
            )
            for key in enc_token_list.keys():
                result[key].append(enc_token_list[key])
        
    return result


def prepare_token_dataset(tokenizer, posting_path: str=None, resume_path: str=None, posting_df: pd.DataFrame=None, resume_df: pd.DataFrame=None, test_size=0.15) -> DatasetDict:
    '''
    Create a Huggingface dataset with train and test splits, containing token datasetets with 'input_ids' and 'attention_mask' columns. 
    Each token sequence has length MAX_SEQ_LEN+2, formed by either padding short texts with EOS tokens or splitting long texts into 
    overlapping sequences of length MAX_SEQ_LEN+2 as done in `group_texts`.
    
    tokenizer: The Huggingface tokenizer to use. Should be pre-trained
    posting_path (optional): Path to the raw CSV file containing posting data. If this is None, then `posting_df` should be provided.
    resume_path (optional): Path to the raw CSV file containing resume data. If this is None, then `resume_df` should be provided.
    posting_df (optional): pandas DataFrame containing raw posting data. If this is None, then `posting_path` should be provided.
    resume_df (optional): pandas DataFrame containing raw resume data. If this is None, then `resume_path` should be provided.
    test_size: Fraction of data to be used as test/val data (float), or number of samples to be used (int), sampled randomly.  
    
    return: Huggingface DatasetDict containing the data, with train and test splits. 
    '''
    if posting_path is None:
        posting_dataset_temp = HFDataset.from_pandas(posting_df)
        posting_dataset = DatasetDict({'train': posting_dataset_temp})
    else:
        posting_dataset = load_dataset('csv', data_files=posting_path)
    
    if resume_path is None:
        resume_dataset_temp = HFDataset.from_pandas(resume_df)
        resume_dataset = DatasetDict({'train': resume_dataset_temp})
    else:
        resume_dataset = load_dataset('csv', data_files=resume_path)
    
    tokenized_posting_dataset = posting_dataset.map(
        preprocess_posting,
        batched=True,
        num_proc=12,
        remove_columns=posting_dataset['train'].column_names,
#         fn_kwargs={'tokenizer': tokenizer}
    )

    tokenized_resume_dataset = resume_dataset.map(
        preprocess_resume,
        batched=True,
        num_proc=12,
        remove_columns=resume_dataset['train'].column_names,
#         fn_kwargs={'tokenizer': tokenizer}
    )
    tokenized_dataset = concatenate_datasets([tokenized_posting_dataset['train'], tokenized_resume_dataset['train']])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=test_size)

    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=12)
    lm_dataset.set_format('torch')
    
    return lm_dataset
    
    
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
    tokenizer = AutoTokenizer.from_pretrained(_EMBED_TOKENIZER)
    # Tokenize with truncation to the model's max length
    tokenized = tokenizer(text, truncation=True, max_length=512, return_tensors='pt')

    model = AutoModel.from_pretrained(_EMBED_MODEL, add_pooling_layer=False, output_hidden_states=True)
    model.eval()

    with torch.no_grad():
#         print(tokenizer.convert_ids_to_tokens(tokenized['input_ids'].flatten()))
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
    
    
def doc_sim_score_with_cl(model, txt1, txt2, device):
    '''
    Get the document similarity score with a model trained with contrastive learning. Similarity is given as vector L2 distance, meaning
    smaller values imply higher similarity.
    
    model: PyTorch model that outputs an embedding for each token (size [n_tokens x embedding_dim]).
    txt1: String representing the first text to compare.
    txt2: String representing the second text to compare.
    device: String or torch.device. The device to use for computations.
    '''
    txt1, txt2 = _clean_text(txt1), _clean_text(txt2)
    tokenizer = AutoTokenizer.from_pretrained(_EMBED_TOKENIZER)
    
    tokenized1 = tokenizer(txt1, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    tokenized2 = tokenizer(txt2, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    
    model.eval()
    with torch.no_grad():
        emb1 = model.get_embedding({k: v.to(device) for k,v in tokenized1.items()})
        emb2 = model.get_embedding({k: v.to(device) for k,v in tokenized2.items()})
    
    
    return torch.linalg.norm(emb1-emb2, ord=2).cpu().item()
    
    

    
def doc_sim_score(emb1: np.ndarray, emb2: np.ndarray, comp_type: str='mean_embeddings_cosine') -> np.ndarray:
    '''
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

        
        
class ContrastiveLearningDataset(Dataset):
    def __init__(self, lm_dataset, mode, tokenizer, augmentation_fns):
        '''
        Create a dataset for a contrastive learning model to use. The dataset will augment each document in the training
        set by creating a positive sample (an augmented view) and a negative sample (a random other document.)
        
        lm_dataset: transformers Dataset as returned by the `prepare_token_dataset` function
        mode: either "train" or "test". Must exist as a split inside of lm_dataset
        tokenizer: Huggingface tokenizer.
        augmentation_fns: List of functions that each accept a document (as str) and returns an augmented view (as str).
        '''
        self.lm_dataset = lm_dataset[mode]
        self.dataset_len = len(self.lm_dataset)
        self.mode = mode
        self.tokenizer = tokenizer
        
        self.augmentation_fns = augmentation_fns
        
        self.build_dataset()
        
        
    @staticmethod
    def augment_txt_parallel(data, tokenizer, augmentation_fns):
        '''under development (parallelizing the augmentation process)'''
        fn = np.random.choice(augmentation_fns)
        raw_txt = tokenizer.decode(data['input_ids'], skip_special_tokens=True)
        aug_txt = fn(raw_txt)
        aug_data = tokenizer(aug_txt, padding='max_length', padding_side='right', truncation=True, max_length=MAX_SEQ_LEN+2)
        return aug_data
    
    @staticmethod
    def build_dataset_parallel(args):
        '''under development (parallelizing the augmentation process)'''
        i, shared_dict = args
        lm_dataset = shared_dict['dataset']
        augment_txt = shared_dict['augment_txt_fn']
        tokenizer = shared_dict['tokenizer']
        augmentation_fns = shared_dict['augmentation_fns']
        
        data = lm_dataset[i]
        pos = augment_txt(data, tokenizer, augmentation_fns)
        neg = lm_dataset[np.random.randint(0, len(lm_dataset))]
        
        return {
                'input_ids': [data['input_ids'], pos['input_ids'][0], neg['input_ids']],
                'attention_mask': [data['attention_mask'], pos['attention_mask'][0], neg['attention_mask']]
            }
        
    
    def augment_txt(self, data):
        '''
        Given a Huggingface data sample as a dict, return an augmented view of the document as a dict.
        
        data: Dict with key "input_ids" (others are not used). 
        return: Dict with keys "input_ids" and "attention_mask", as returned by tokenizer.__call__.
        '''
        fn = np.random.choice(self.augmentation_fns)
        raw_txt = self.tokenizer.decode(data['input_ids'], skip_special_tokens=True)
        aug_txt = fn(raw_txt)
        aug_data = self.tokenizer(aug_txt, padding='max_length', padding_side='right', truncation=True, max_length=512, return_tensors='pt')
        return aug_data
    
    
    def build_dataset(self):
        '''
        Build the triplet dataset of (anchor, positive, negative) samples by creating one sample for each document. 
        Sets self.cur_data as the current sampled dataset to be used during training. This is numpy array containing Hugginface
        BatchEncodings, each with "input_ids" and "attention_mask" each of size [3 x (MAX_SEQ_LEN+2)]. Therefore, if using a 
        DataLoader, each batched input will have size [batch_size x 3 x (MAX_SEQ_LEN+2)]. 
        
        WARNING: Can take a while to run! 
        
        return: None
        '''
#         with Manager() as manager:
#             shared_dict = manager.dict()
#             shared_dict['dataset'] = self.lm_dataset
#             shared_dict['augment_txt_fn'] = ContrastiveLearningDataset.augment_txt_parallel
#             shared_dict['tokenizer'] = self.tokenizer
#             shared_dict['augmentation_fns'] = self.augmentation_fns
            
#             with Pool(12) as pool:
# #                 args = list(zip(range(len(self)), [self] * len(self)))
#                 args = [(i, shared_dict) for i in range(len(self))]
#                 print('Beginning')
#                 res = list(tqdm(pool.imap(ContrastiveLearningDataset.build_dataset_parallel, args), total=len(self)))
#                 self.cur_data = res
        self.cur_data = []
        for i in tqdm(list(range(len(self)))):
            # i is the index of the "anchor"
            data = self.lm_dataset[i]
            pos = self.augment_txt(data)
            neg = self.lm_dataset[np.random.randint(0, self.dataset_len)]
            
            # (anchor, positive sample, negative sample)
            self.cur_data.append(BatchEncoding({
                'input_ids': torch.stack([data['input_ids'].squeeze(), pos['input_ids'].squeeze(), neg['input_ids'].squeeze()]),
                'attention_mask': torch.stack([data['attention_mask'].squeeze(), pos['attention_mask'].squeeze(), neg['attention_mask'].squeeze()])
            }))
            
            
    def __len__(self):
        return self.dataset_len
    
    
    def __getitem__(self, idx):
        return self.cur_data[idx]
        
        

class ContrastiveLearningModel(nn.Module):
    def __init__(self, bert_model, out_embed_dim=512):
        '''
        Model to be trained with constrastive learning with triplet loss.
        
        bert_model: The pretrained BERT or BERT-based model that serves as pretrained embeddings. Will be frozen during training.
        out_embed_dim: Dimension of the output embeddings per token.
        '''
        super().__init__()
        
        self.bert_model = bert_model
        
        for param in self.bert_model.parameters():
            param.requires_grad = False
            
            
        self.layers = nn.Sequential(
            nn.Linear(768*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_embed_dim),
        )
        
    
    def forward(self, x):
        '''
        Runs the forward pass, 
        x should be a BatchEncoding with input_ids and attention_mask as pytorch format. Each tensor should be size 
        [batch_size, n_tokens].
        '''
        out_hid = self.bert_model(**x).hidden_states
        out_concat = torch.cat(out_hid[-4:], dim=2)[:, 1:-1].mean(dim=1)
        return self.layers(out_concat)
        
        
        
def train_model(model, optimizer, dataset, loss_fn, epochs, batch_size, save_freq=None, save_path=None, scheduler=None, device='cpu'):
    '''
    Train a model for contrastive learning.
    
    model: Model to train.
    optimizer: Initialized optimizer
    dataset: ContrastiveLearningDataset
    loss_fn: A loss function suitable for contrastive learning (e.g. TripletMarginLoss)
    epochs: Number of epochs to train for
    batch_size: Batch size to train with
    save_freq: Number of epochs in between saves (model and optimizer will be saved to disk)
    save_path: Path to save model to.
    scheduler: Learning rate scheduler.
    device: The device to train on. Defaults to 'cpu'. Set to 'cuda:0' to use a GPU. 
    '''
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Split a single BatchEncoding (that has data [batch_size x 3 x 512]) into three separate BatchEncodings for anchor, pos, and neg
    split_triplet_data = lambda x: (BatchEncoding({k: v[:, i] for k,v in x.items()}) for i in range(3))
    
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    print(f'Scheduler: {scheduler}' if scheduler else 'No learning rate scheduling!')
    print(f'Training for {epochs} epochs, with batch size={batch_size}')
    print(f'Using device: {device}')
    print(f'Saving model every {save_freq} epochs to {save_path}' if save_freq and save_path else 'WARNING: Will not save model!')

    for e in range(epochs):
        losses = []
        t = time.time()
        
        print(f'\n-----Epoch {e+1}/{epochs}-----')
        for i, x in enumerate(loader):
            # Split the batch into the anchor, positive, and negative parts, and run them through the model independently
            data_enc, pos_enc, neg_enc = split_triplet_data(x)
            data, pos, neg = model(data_enc.to(device)), model(pos_enc.to(device)), model(neg_enc.to(device))
            loss = loss_fn(data, pos, neg)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

            if len(losses) == 50 or i == len(loader)-1:
                elapsed = time.time() - t
                t = time.time()
                print(f'Batch {i+1}/{len(loader)}, loss: {np.mean(losses)} ({elapsed:.3f}s)')
                losses = []
                
        # Rebuild the dataset; i.e. re-sample the anchor augmentations and negative samples
        dataset.build_dataset()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if scheduler is not None:
            scheduler.step()
            
        if save_freq and save_path and ((e+1) % save_freq == 0 or e == epochs-1):
            save_model(save_path, model, optimizer, epochs)
            print(f'Saved to {save_path}')       
            
            
def save_model(save_path, model, optimizer, epoch):
    '''
    Save a model to disk, with extra training parameters in order to resume training later.
    
    save_path: Path as string to location to save the model.
    model: Model to save.
    optimizer: Optimizer to save.
    epoch: The last trained epoch (for info only; not used when resuming training)
    '''
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 
        save_path)
    
    
def load_model(model, save_path, strict=True):
    '''
    Load a previously saved model for inference.
    
    model: The PyTorch model to load weights into
    save_path: Path as string to the model (.pth)
    strict: Whether to strictly load weights (kwarg for load_state_dict)
    '''
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    return model