import numpy as np
from transformers import (
    RobertaTokenizer, 
    RobertaForMaskedLM, 
    DataCollatorForLanguageModeling, 
    TrainingArguments,
    Trainer,
)

from datasets import load_dataset, concatenate_datasets
from embeddings.contrastive_learning import prepare_token_dataset

BLOCK_SIZE = 128 # Stride length when splitting long texts into 512-length segments
MAX_SEQ_LEN = 512 # maximum length of a sequence that BERT can operate on




if __name__ == '__main__':
    posting_csv_path = './temp/postings5k.csv'
    resume_csv_path = './resume_data/Resume.csv'
    posting_txt_col = 'description'
    resume_txt_col = 'Resume_str'
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    
    lm_dataset = prepare_token_dataset(tokenizer, posting_path=posting_csv_path, resume_path=resume_csv_path)
    
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True
    )
    
    training_args = TrainingArguments(
        output_dir="./temp/roberta-tuned",
        overwrite_output_dir=True,
        logging_strategy='epoch',
        eval_strategy='epoch',
        num_train_epochs=25,
    #     learning_rate=2e-5,
        per_device_train_batch_size=48,
    #     save_steps=500,
        save_strategy='epoch',
        save_total_limit=1,
        seed=1
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lm_dataset['train'],
        eval_dataset=lm_dataset['test']
    )
    
    trainer.train()

    trainer.save_model('./roberta-tuned')