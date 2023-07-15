'''
generates tweets based on the fine-tuned gpt-neo model
'''
import os, ndjson
from typing import Optional

import torch
from transformers import GPT2Tokenizer

def generate_tweets(label: str, num_seq: int, prompt: Optional[str]="<|startoftext|>"):
    '''
    function for generating tweets using the fine-tuned gpt-neo model

    Args:
        label (str): the label for the name of the model
        num_seq (int): number of sequences (i.e. tweets) the model will generate
    '''
    model = torch.load(f'results/{label}_model.pt')

    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B",    
                                    bos_token="<|startoftext|>",
                                    eos_token="<|endoftext|>",
                                    pad_token="<|pad|>")

    # generate tweets
    print('generate tweets')
    generated = tokenizer(prompt,   
                        return_tensors="pt").input_ids.cuda()
    sample_outputs = model.generate(generated, 
                    # Use sampling instead of greedy decoding 
                    do_sample=True, 
                    # Keep only top 50 token with 
                    # the highest probability
                    top_k=50, 
                    # Maximum sequence length
                    max_length=240, 
                    # Keep only the most probable tokens 
                    # with cumulative probability of 95%
                    top_p=0.95, 
                    # Changes randomness of generated sequences
                    temperature=1.9,
                    # Number of sequences to generate                 
                    num_return_sequences=num_seq)
    
    generated_tweets = []
    for output in sample_outputs:
            generated_tweet = tokenizer.decode(output, skip_special_tokens=True)
            generated_tweets.append(generated_tweet)
    return generated_tweets
    

def main(labels, num_seq, prompt):
    print(f'>>>>>> Prompt is: "{p}" <<<<<<')

    for label in labels:
        print(f'Label is {label} \n ------ \n')
        generated_tweets = generate_tweets(label, num_seq, prompt)

        # save
        if not os.path.exists('generated_tweets'):
            os.makedirs('generated_tweets')
        save_path = os.path.join('generated_tweets', f'{label}_tweets.ndjson')
        for i, tweet in enumerate(generated_tweets):
            print(f'{i}: {tweet} \n ------ \n')

        line = [{'prompt': prompt, 'generated_text': generated_tweets}]
        with open(save_path, 'a') as f:
            ndjson.dump(line, f)
            f.write('\n')



if __name__=='__main__':
    labels = ['POS', 'NEG', 'NEU']
    prompts = ["Denmark", 
               "Denmark has lifted all covid restrictions",
               "Denmark's covid policy is",
               "Covid infections in Denmark",
               "Covid restrictions in Denmark",
               "Covid deaths in Denmark",
               "It looks like Denmark is",
               "Why is Denmark"]

    for p in prompts:
        main(labels, num_seq=20, prompt=p)

