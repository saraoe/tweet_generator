'''
obtaining the tweets for fine-tuning of gpt-neo
'''

import pandas as pd
import os, ndjson
from glob import glob

def ndjson_gen(path):
    for in_file in glob(path):
        with open(in_file) as f:
            reader = ndjson.reader(f)

            for post in reader:
                yield post 

def write_line(path, line):
    with open(path, 'a') as f:
            ndjson.dump(line, f)
            f.write('\n')

def main(cls_path, tweets_path, out_path, labels):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    df = pd.read_csv(cls_path, lineterminator='\n')

    for i, post in enumerate(ndjson_gen(tweets_path)):
        for label in labels:
            tmp_df = df[df['polarity']==label]
            if int(post['id']) in list(tmp_df['id']): # check if id is in relevant df
                # write to ndjson
                line = [{'id': post['id'], 'text': post['text']}]
                path = os.path.join(out_path, f'tweets_polarity_{label}.ndjson')
                write_line(path,line)
                print(f'line saved in {path}')
        if i % 100 == 0:
            print('---------', f'Tweet {i} done', '---------', sep = '\n')


if __name__=='__main__':
    cls_path = os.path.join('..', 'HOPE-keyword-query-Twitter', 'denmark_files', 'denmark_final.csv')
    tweets_path = os.path.join('/data', 'twitter-omicron-denmark', '*')
    out_path = os.path.join('/home', 'saram', 'data', 'gpt-neo')
    
    main(cls_path, tweets_path, out_path, labels=['NEG', 'NEU', 'POS'])


