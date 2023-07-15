'''
fine-tuning gpt-neo on tweets
'''
import os, ndjson
from glob import glob

import torch
from torch.utils.data import Dataset,  random_split
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments

def ndjson_gen(path):
    for in_file in glob(path):
        with open(in_file) as f:
            reader = ndjson.reader(f)

            for post in reader:
                yield post 


class TweetDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length) -> None:
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer(
                f"<|startoftext|>{txt}<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
            input_ids = torch.tensor(encodings_dict['input_ids'])
            self.input_ids.append(input_ids)
            mask = torch.tensor(encodings_dict['attention_mask'])
            self.attn_masks.append(mask)

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.attn_masks[index]


def train(path: str, label: str):
    '''
    function for fine-tuning gpt-neo on tweets

    Args:
        path (str): path for ndjson file of tweets (text must be in column 'text')
        label (str)
    '''
    print(f'>>> Label = {label}')

    texts = []
    for post in ndjson_gen(path):
        texts.append(post['text'])

    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").cuda()
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B",    
                                bos_token="<|startoftext|>",
                                eos_token="<|endoftext|>",
                                pad_token="<|pad|>")
    model.resize_token_embeddings(len(tokenizer))

    max_length = max([len(tokenizer.encode(text)) for text in texts])

    # initialize dataset
    print('>> initialize dataset')
    dataset = TweetDataset(texts, tokenizer, max_length)

    train_size = int(0.9 * len(dataset))
    upper = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size,upper])
    print('>> done splitting data')

    training_args = TrainingArguments(output_dir='results',
                                    num_train_epochs=2,
                                    logging_strategy='epoch',
                                    load_best_model_at_end=True,
                                    save_strategy='epoch',
                                    evaluation_strategy='epoch',
                                    per_device_train_batch_size=2,
                                    per_device_eval_batch_size=2,
                                    warmup_steps=100, weight_decay=0.01,
                                    logging_dir='logs')

    trainer = Trainer(model=model, args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    data_collator=lambda data: 
                        {"input_ids": torch.stack([f[0] for f in data]),       
                        "attention_mask": torch.stack([f[1] for f in data]),
                        "labels": torch.stack([f[0] for f in data])})
    print('>> ready for training')
    trainer.train() # start training
    print('>> done training')
    torch.save(model, os.path.join('results', f'{label}_model.pt'))


if __name__=='__main__':
    path = os.path.join('/home', 'saram', 'data', 'gpt-neo')
    labels = ['POS', 'NEG', 'NEU']

    for label in labels:
        label_path = os.path.join(path, f'tweets_polarity_{label}.ndjson')
        train(label_path, label)
    