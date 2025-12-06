import numpy as np
import pandas as pd
import re
import torch

from collections.abc import Callable
from multiprocessing import Pool
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from transformers import (
    RobertaTokenizer, 
    RobertaModel,
    RobertaForMaskedLM,
    BatchEncoding
)

# Tokenizer (will be downloaded from Huggingface)
TOKENIZER = RobertaTokenizer.from_pretrained('roberta-base')
# Path to the tuned bert model that serves as the pretrained layers for the CL model
try:
    BASE_MODEL = RobertaModel.from_pretrained('./roberta-tuned-v1', add_pooling_layer=False, output_hidden_states=True)
except:
    # If no tuned model exists, just use RoBERTA base as the base model
    BASE_MODEL = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False, output_hidden_states=True)

BLOCK_SIZE = 128 # Stride length when splitting long texts into 512-length segments
MAX_SEQ_LEN = 510 # maximum length of a sequence that BERT can operate on (512 = 510 tokens + start token + end token)

posting_txt_col = 'description'
resume_txt_col = 'Resume_str'

DEVICE = (f'cuda:0' if torch.cuda.is_available() else 'cpu')
if DEVICE == 'cpu':
    print('WARNING! Training PyTorch models on CPU may result in extremely long training time!')



'''
Document similarity functions: Methods to compute the similarity between two documents at both the token level and 
document level.
'''

def soft_alignment_similarity(anchors: torch.Tensor, others: torch.Tensor, anc_mask: torch.Tensor, ot_mask: torch.Tensor, temperature: float=0.1) -> torch.Tensor:
    """
    Compute the soft alignment similarity between an anchor document and another document given their embeddings. The similarity is not symmetric, i.e., sim(A,B) =/= sim(B,A). 
    The comparison is done at a token-level. Higher score implies higher similarity. This function is designed to be called on batches of documents, and will ignore 
    padding or separation tokens as given in the attention masks `anc_mask` and `ot_mask`. The documents should be embedding tensors of shape [B, MAX_SEQ_LEN+2, D] where
    B is the batch size and D is the token embedding dimension. Each document on the anchor batch is compared to its corresponding document in the other batch.
    
    anchors: Anchor documents batch of size [B, 512, D]
    others: Other documents to compare to of size [B, 512, D]
    anc_mask, ot_mask: [B, 512] (1 for valid tokens, 0 for pad/special tokens). The mask tensors of the anchors and others, respectively.
    temperature: Hyperparameter between 0 and 1. Higher values result in a smoother average over tokens, while lower values focus on highest similarity between
                 any two token pairs. Recommended: 0.05 to 0.1. Default: 0.1.
    
    returns: Similarity scores for the batch in a B-dimensional vector.
    """

    B, L, D = anchors.shape
    sims = torch.einsum("bid,bjd->bij", anchors, others) / temperature  # [B, L, L]

    # Mask out padding tokens
    ot_mask = ot_mask.bool().unsqueeze(1)           # [B, 1, L]
    anc_mask = anc_mask.bool().unsqueeze(2)           # [B, L, 1]

    # Replace padding positions in B with -inf before logsumexp
    sims = sims.masked_fill(~ot_mask, float("-inf"))

    # logsumexp over B’s tokens
    logsumexp = torch.logsumexp(sims, dim=2)  # [B, L]

    # Mask out padding tokens in A (so we don't average over them)
    logsumexp = logsumexp.masked_fill(~anc_mask.squeeze(2), 0.0)

    # Compute average only over valid A tokens
    valid_counts = anc_mask.squeeze(2).sum(dim=1)  # [B]
    sim_per_doc = logsumexp.sum(dim=1) / valid_counts
    
    return sim_per_doc


def word_movers_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    '''
    DEPRECATED. NOT UP-TO-DATE.
    Compute the Word Mover's Distance (WMD) between documents, each given as arrays of token embeddings. 
    
    emb1, emb2: Embeddings of the documents to compute, each as ndarray of shape [n_tokens, D]. D is the embedding dimension. The documents
                may have a different number of tokens.
    '''
    # Compute pairwise distance matrix between token embeddings
    M = ot.dist(emb1, emb2, metric='euclidean')
    M /= M.max()  # Normalize for stability

    # Uniform weights (bag-of-words)
    a = np.ones(emb1.shape[0]) / emb1.shape[0]
    b = np.ones(emb2.shape[0]) / emb2.shape[0]

    # Compute Earth Mover's Distance (Wasserstein-1)
    distance = ot.emd2(a, b, M)
    return distance


def cosine_similarity(arr1: np.ndarray, arr2: np.ndarray) -> float:
    '''
    Return the cosine similarity between two vectors.
    
    arr1, arr2: 1D ndarrays of the same shape. 
    '''
    return (np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2)))


def embedding_mean(emb: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    '''
    Takes the mean embeddings of the per batch given a batched tensor of embeddings, while ignoring
    padding and special token embeddings. .
    
    emb: Batched embeddings of shape [batch_size, max_n_tokens, embedding_dim].
    attn_mask: Binary mask for each document of shape [batch_size, max_n_tokens]. 0 represents a token to ignore
    
    Returns a tensor of size [batch_size x embedding_dim], the mean embedding vector for each document in the batch, with pad/special tokens ignored.
    '''
    return torch.where(attn_mask.to(bool).unsqueeze(2), emb, torch.nan).nanmean(dim=1)
    
    
'''
Augmentation functions: For creating a postive view of a document during contrastive learning
'''

def split_sentences(text: str) -> str:
    # Convenience function to split text around punctuation
    # Keeps newlines intact but splits on ., ?, !
    text = text.replace("\n", " <N> ")  # temporary marker
    sents = re.split(r'(?<=[.!?])*', text)
    sents = [s.replace("<N>", "\n") for s in sents]
    return [s.strip() for s in sents if s.strip()]


def random_mask(text: str, mask_prob: float=0.25) -> str:
    # Randomly mask a percentage of tokens for MLM-type contrastive learning
    tokens = TOKENIZER.tokenize(text)
    new_tokens = []
    for t in tokens:
        if np.random.random() < mask_prob:
            new_tokens.append(TOKENIZER.mask_token)
        else:
            new_tokens.append(t)
    return TOKENIZER.convert_tokens_to_string(new_tokens)


def sentence_dropout(text: str, drop_prob: float=0.35) -> str:
    # Randomly drop 35% of sentences from the document
    sentences = split_sentences(text)
    keep = [s for s in sentences if np.random.random() > drop_prob]
    if not keep:  # Ensure at least one sentence
        try:
            keep = np.random.choice(sentences, 1)
        except:
            print(text)
    return " ".join(keep)


def sentence_shuffle(text: str) -> str:
    # Randomly shuffle the sentences in the document
    sentences = split_sentences(text)
    np.random.shuffle(sentences)
    return " ".join(sentences)


def random_span_deletion(text: str, num_spans: int=16, span_length: int=4) -> str:
    # Randomly delete several contiguous spans of tokens from the document
    tokens = TOKENIZER.tokenize(text)
    n = len(tokens)
    for _ in range(num_spans):
        if n <= span_length:
            break
        start = np.random.randint(0, max(0, n - span_length))
        del tokens[start:start+span_length]
        n = len(tokens)
    return TOKENIZER.convert_tokens_to_string(tokens)




class ContrastiveLearningDataset(Dataset):
    def __init__(self, lm_dataset, mode, tokenizer, augmentation_fns, num_sample=None):
        '''
        Create a dataset for a contrastive learning model to use. The dataset will augment each document in the training
        set by creating a positive sample (an augmented view) and a negative sample (a random other document.)
        
        lm_dataset: transformers Dataset as returned by the `prepare_token_dataset` function
        mode: either "train" or "test". Must exist as a split inside of lm_dataset
        tokenizer: Huggingface tokenizer.
        augmentation_fns: List of functions that each accept a document (as str) and returns an augmented view (as str).
        num_sample: Number of documents to sample for each epoch. If given, the dataset will sample from `lm_dataset` every epoch
                    and train on that sampled data for the epoch. If None, use the entire `lm_dataset` each epoch. Default: None
        ''' 
        self.lm_dataset = lm_dataset[mode]
        if num_sample > len(self.lm_dataset):
            raise ValueError(f'num_sample is greater than size of original dataset ({num_sample} > {len(lm_dataset)})')
            
        self.dataset_len = len(self.lm_dataset) if num_sample is None else num_sample
        self.mode = mode
        self.tokenizer = tokenizer
        self.num_sample = num_sample
        
        self.augmentation_fns = augmentation_fns
        
        self.build_dataset()
        
        
    @staticmethod
    def augment_txt_parallel(args: tuple[str, Callable]) -> str:
        '''
        Augment a document by calling an augmentation function on it. Designed to be called from multiprocessing.Pool.
        
        args: 2-tuple of (raw_txt, fn). The text and the function to call on it.
        '''
        raw_txt, fn = args
        aug_txt = fn(raw_txt)
        aug_data = TOKENIZER(aug_txt, padding='max_length', padding_side='right', truncation=True, max_length=MAX_SEQ_LEN+2, return_tensors='pt')
        return aug_data
    
    
    def augment_txt(self, data: dict) -> str:
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
        
        Can optionally sample a subset of the dataset if `self.num_sample` is set.
        
        WARNING: Can take a while to run! 
        
        return: None
        '''
        self.cur_data = []
        chunk_size = 750
        
        ind = np.random.randint(0, len(self.lm_dataset), size=self.num_sample)
        dataset = (self.lm_dataset if self.num_sample is None else self.lm_dataset.select(ind))

        with Pool() as pool:
            # Process in chunks
            for i in (list(range(0, self.dataset_len, chunk_size))):
                data = dataset[i : i+chunk_size]
                raw_txt_list = TOKENIZER.batch_decode(data['input_ids'], skip_special_tokens=True)
                args = [(txt, np.random.choice(self.augmentation_fns)) for txt in raw_txt_list]
                pos = list(pool.map(ContrastiveLearningDataset.augment_txt_parallel, args, chunksize=15))
                
                neg = self.lm_dataset[np.random.randint(0, len(self.lm_dataset), size=len(raw_txt_list))]
                
                self.cur_data.extend([
                    BatchEncoding({
                        'input_ids': torch.stack([data['input_ids'][j], pos[j]['input_ids'].squeeze(), neg['input_ids'][j]]),
                        'attention_mask': torch.stack([data['attention_mask'][j], pos[j]['attention_mask'].squeeze(), neg['attention_mask'][j]]),

                    })
                    for j in range(len(raw_txt_list))
                ]) 
                
            
    def __len__(self):
        return self.dataset_len
    
    
    def __getitem__(self, idx):
        return self.cur_data[idx]
    
    
def preprocess_posting(x: str) -> BatchEncoding:
    '''Preprocess for a job listing, as read from the linkedin job posting CSV file'''
    return TOKENIZER(x[POSTING_TXT_COL], add_special_tokens=False)


def preprocess_resume(x: str) -> BatchEncoding:
    '''Preprocess for a resume CSV file, as read from the resume CSV file'''
    return TOKENIZER(x[RESUME_TXT_COL], add_special_tokens=False)


def group_texts(token_data: dict, return_tensors: str|None=None) -> dict[str, list]:
    '''
    Split texts that are two long (longer than 512 tokens) into overlapping segments with a sliding window of stride BLOCK_SIZE,
    and window size of MAX_SEQ_LEN+2. For example, for a BLOCK_SIZE of 128 and MAX_SEQ_LEN of 510, a text with 1376 tokens is broken 
    down into 8 segments that start at the following indices:
        [0, 128, 256, 384, 512, 640, 768, 896]
    Then each segment has size:
        [512, 512, 512, 512, 512, 512, 512, 480]
        
    token_data: Dict-like (such as BatchEncoding or standard dict) with keys {'input_ids', 'attention_mask'} representing a batched
                list of tokenized documents.
    return_tensors: Same as `return_tensors` argument in tokenizer.__call__
    '''
    result = {k: [] for k in token_data.keys()}
    all_keys, all_data = zip(*list(token_data.items()))

    for data_tuple in zip(*all_data):
        # Data tuple consists of one list for each key in the data (should just be 'input_ids' and 'attention_mask' of equal length)         
        # Not a general assumption that this is what is inside data_tuple
        token_list, attn_mask = data_tuple
        n_tokens = len(token_list)
        shifts = [t for t in range(0, n_tokens, BLOCK_SIZE) if t+MAX_SEQ_LEN+2-n_tokens < BLOCK_SIZE]
        if len(shifts) == 0:
            shifts = [0]
        for shift in shifts:
            # Re-encode the token list with the special eos/bos characters
            enc_token_list = TOKENIZER(
                TOKENIZER.decode(token_list[shift : shift+MAX_SEQ_LEN]), 
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


def prepare_token_dataset(posting_path: str=None, resume_path: str=None, posting_df: pd.DataFrame=None, resume_df: pd.DataFrame=None, test_size=0.2) -> DatasetDict:
    '''
    Create a Huggingface dataset with train and test splits, containing token datasetets with 'input_ids' and 'attention_mask' columns. 
    Each token sequence has length MAX_SEQ_LEN+2, formed by either padding short texts with EOS tokens or splitting long texts into 
    overlapping sequences of length MAX_SEQ_LEN+2 as done in `group_texts`.
    
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
        num_proc=24,
        remove_columns=posting_dataset['train'].column_names,
    )

    tokenized_resume_dataset = resume_dataset.map(
        preprocess_resume,
        batched=True,
        num_proc=24,
        remove_columns=resume_dataset['train'].column_names,
    )
    tokenized_dataset = concatenate_datasets([tokenized_posting_dataset['train'], tokenized_resume_dataset['train']])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=test_size)

    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=12)
    lm_dataset.set_format('torch')
    
    return lm_dataset
        
        

class ContrastiveLearningModel(nn.Module):
    def __init__(self, bert_model: nn.Module, out_embed_dim: int=588):
        '''
        Model to be trained with constrastive learning with triplet loss. The model learns by first creating augmented views of a document, 
        such as rearranging sentences, dropping sentences, randomly masking tokens, etc. It then trains minimize the distance between a
        document (anchor) and any augmented view of it (positive sample), while maximizing the distance between the document and some other 
        randomly selected document (negative sample).
        
        bert_model: The pretrained BERT or BERT-based model that serves as pretrained embeddings. Will be fine-tuned during training.
        out_embed_dim: Dimension of the output embeddings per token.
        '''
        super().__init__()
        
        self.bert_model = bert_model
        
#         for param in self.bert_model.parameters():
#             param.requires_grad = False
            
            
        self.layers = nn.Sequential(
#             nn.Linear(768*4, 1024),
#             nn.ReLU(),
            nn.Linear(768*4, out_embed_dim)
        )
        
    
    def forward(self, x: BatchEncoding):
        '''
        Runs the forward pass, 
        x should be a BatchEncoding with input_ids and attention_mask as pytorch format. Each tensor should be size 
        [batch_size, n_tokens].
        '''
        out_hid = self.bert_model(**x).hidden_states
        out_concat = torch.cat(out_hid[-4:], dim=2)#.mean(dim=1)
        return self.layers(out_concat)
        

def soft_alignment_loss(anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, 
                        anc_mask, pos_mask, neg_mask,
                        normalize=True, margin=5.) -> torch.Tensor:
    '''
    Contrastive learning loss function using token-token soft alignment. Symmetric similarity is used, i.e. S = 0.5 * (sim(A,B) + sim(B,A)).
    
    anchor, pos, neg: Tensors of size [batch_size, MAX_SEQ_LEN+2, embedding_dim]. The anchor, positive, and negative samples.
    anc_mask, pos_mask, neg_mask: Binary tensors of size [batch_size, MAX_SEQ_LEN+2]. The attention masks for the anchor, positive, and negative samples.
    normalize: If True, normalize all token embedding vectors. Default: True.
    margin: Contrastive learning margin. Default: 5.0
    '''
    
    if normalize:
        anchor = F.normalize(anchor, dim=2)
        pos = F.normalize(pos, dim=2)
        neg = F.normalize(neg, dim=2)
    
    s_pos = 0.5 * (
        soft_alignment_similarity(anchor, pos, anc_mask, pos_mask) +
        soft_alignment_similarity(pos, anchor, pos_mask, anc_mask)
    )
    s_neg = 0.5 * (
        soft_alignment_similarity(anchor, neg, anc_mask, neg_mask) +
        soft_alignment_similarity(neg, anchor, neg_mask, anc_mask)
    )
    
    return F.relu(margin - s_pos + s_neg).mean()
        
        



def train_model(model: nn.Module, optimizer: optim.Optimizer, dataset: Dataset, loss_fn: Callable, epochs: int, batch_size: int, 
                save_freq: int|None=None, save_path:str|None=None, scheduler=None, device: str='cpu'):
    '''
    Train a model for contrastive learning.
    
    model: Model to train.
    optimizer: Initialized optimizer
    dataset: Initialized ContrastiveLearningDataset
    loss_fn: A loss function suitable for contrastive learning (e.g. soft_alignment_loss, nn.TripletMarginLoss)
    epochs: Number of epochs to train for
    batch_size: Batch size to train with
    save_freq: Number of epochs in between saves (model and optimizer will be saved to disk), or None for no saving. Default: None
    save_path: Path to save model to, or None for no saving. Default: None
    scheduler: Optional learning rate scheduler. Default: None
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
            optimizer.zero_grad()
            # Split the batch into the anchor, positive, and negative parts, and run them through the model independently
            data_enc, pos_enc, neg_enc = split_triplet_data(x)
            data, pos, neg = model(data_enc.to(device)), model(pos_enc.to(device)), model(neg_enc.to(device))
            
            loss = loss_fn(data, pos, neg, 
                           data_enc['attention_mask'],
                           pos_enc['attention_mask'],
                           neg_enc['attention_mask']
                          )
            
            
#             loss = loss_fn(embedding_mean(data, data_enc['attention_mask']), 
#                            embedding_mean(pos, pos_enc['attention_mask']), 
#                            embedding_mean(neg, neg_enc['attention_mask']))

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if len(losses) == 50 or i == len(loader)-1:
                elapsed = time.time() - t
                t = time.time()
                print(f'Batch {i+1}/{len(loader)}, loss: {np.mean(losses)} ({elapsed:.3f}s)')
                losses = []
        
        if scheduler is not None:
            scheduler.step()
            
        if save_freq and save_path and ((e+1) % save_freq == 0 or e == epochs-1):
            save_model(save_path, model, optimizer, epochs)
            print(f'Saved to {save_path}')   
            
        # Rebuild the dataset; i.e. re-sample the anchors, augmentations (positive), and negative samples
        dataset.build_dataset()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            
def save_model(save_path: str, model: nn.Module, optimizer: optim.Optimizer, epoch: int):
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
    
    
def load_model(model: nn.Module, save_path: str, strict: bool=True):
    '''
    Load a previously saved model for inference.
    
    model: The PyTorch model to load weights into
    save_path: Path as string to the model (.pth)
    strict: Whether to strictly load weights (kwarg for load_state_dict)
    '''
    checkpoint = torch.load(save_path, weights_only=True, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    return model


def fetch_cl_model(model_path: str, out_embed_dim: int=588, device: str='cpu'):
    model = ContrastiveLearningModel(BASE_MODEL, out_embed_dim=out_embed_dim)
    model = load_model(model, model_path).to(device)
    return model


AUGMENTATION_FNS = [
    random_mask,
    sentence_dropout,
    sentence_shuffle,
    random_span_deletion,
]


if __name__ == '__main__':
    # Example on how to train a CL model
    
    # Uncomment these 3 lines to re-process and save the lm dataset:
#     posting_df = pd.read_csv('/home/hice1/khom9/scratch/CS6220_Project/postings.csv').dropna(subset=['description'])
#     lm_dataset = prepare_token_dataset(tokenizer, posting_df=posting_df, resume_path='./resume_data/Resume.csv')
#     lm_dataset.save_to_disk('./temp/lm_dataset')
    
    # Load the previously processed lm dataset
    lm_dataset = load_from_disk('./temp/lm_dataset')
    dataset = ContrastiveLearningDataset(lm_dataset, 'train', TOKENIZER, AUGMENTATION_FNS, num_sample=5000)
    
    
    batch_size = 32
    lr = 1e-5
    m = ContrastiveLearningModel(BASE_MODEL, out_embed_dim=588).to(DEVICE)
    optimizer = optim.Adam(m.parameters(), lr=lr)
    loss_fn = soft_alignment_loss #nn.TripletMarginLoss(margin=1.5) 
    save_path = './temp/contrastive_learning.pth'

    epochs = 200

    train_model(m, optimizer, dataset, loss_fn, epochs, batch_size, device=DEVICE, save_path=save_path, save_freq=1)
