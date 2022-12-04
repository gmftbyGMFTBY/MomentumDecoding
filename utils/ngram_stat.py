import json
import ipdb
from tqdm import tqdm
import torch
import nltk

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def extract_ngrams(content, ngram_list):
    tokens = nltk.word_tokenize(content)
    ngrams = set()
    in_num, out_num = 0, 0
    result = {i:[0,0] for i in [2,3,4,5,6,7,8]}
    for i in range(len(tokens)):
        for n in ngram_list:
            if i - n >= 0:
                ngram = tokens[i-n:i]
                if ' '.join(ngram) in ngrams:
                    result[n][0] += 1
                else:
                    result[n][1] += 1
                ngrams.add(' '.join(ngram))
    return result

if __name__ == '__main__':
    for dataset in ['story', 'wikitext', 'wikinews']:
        with open(f'{dataset}/{dataset}.jsonl') as f:
            counter = 0
            results = {i:[0,0] for i in [2,3,4,5,6,7,8]}
            for line in tqdm(f.readlines()):
                line = json.loads(line)['text'].strip()
                result = extract_ngrams(line, [2, 3, 4, 5, 6, 7, 8])
                for key, value in result.items():
                    results[key][0] +=result[key][0]
                    results[key][1] +=result[key][1]
        for key, value in results.items():
            print(f'[!] {key}-gram: {round(value[0]/(value[0] + value[1]), 4)}')
