from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from transformers import BertTokenizer, BertModel

from database import Posting


def fetch_postings_text(database_url):
    '''
    Returns a list of the postings' body text from the database
    '''
    engine = create_engine(database_url)
    with Session(engine) as session:
        stmt = select(Posting.description)
        return list(session.scalars(stmt))
    
    
def embed_text(text: str, word_reduction_type: str='sum_last_four') -> np.ndarray:
    '''
    Get array of word embeddings for a single document (text). Word embeddings can be computed from BERT's layer outputs
    in several ways (word_reduction_type):
    - sum_last_four: Sum the outputs of the last four layers
    - concat_last_four: Concatenate the outputs of the last four layers in one long vector (NOT YET IMPLEMENTED)
    - last_layer: Use only the lasy layer as the embedding (NOT YET IMPLEMENTED)
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens_t = torch.tensor([tokenizer.encode(text)])
    segment_t = torch.tensor([[1] * tokens_t.shape[1]])
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()
    with torch.no_grad():
        out = model(tokens_t, segment_t)
        hidden_states = out[2]
        hidden_states = torch.cat(hidden_states).permute(1, 0, 2)
        
        if reduction_type == 'sum_last_four':
            embeddings = hidden_states[:, -4:].sum(dim=1)
        
        return embeddings
