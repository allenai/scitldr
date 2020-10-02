""" Script to calculate the statistics of the generated tldrs """

from argparse import ArgumentParser
import glob
import pandas as pd
from multiprocessing.pool import Pool
from tqdm.auto import tqdm
import json
import nltk
from nltk.util import ngrams
import numpy as np


def get_novelty(summ, text, ngram=1):
    summ_tokens = list(ngrams(summ, ngram))
    text_tokens = ngrams(text, ngram)
    all_tokens = set(text_tokens)
    res = len([e for e in summ_tokens if e not in all_tokens]) / len(summ_tokens) if len(summ_tokens) > 0 else 0
    return res
    

def word_tokenize(text):
    return text.split()


def get_stats(dataset_instance):
    """ dataset_instance is a dict with two fields `summary` and `text` """
    summ_tokens = word_tokenize(dataset_instance['summary'])
    text_tokens = word_tokenize(dataset_instance['text'])
    novelty1 = get_novelty(summ_tokens, text_tokens, 1)
    novelty2 = get_novelty(summ_tokens, text_tokens, 2)
    novelty3 = get_novelty(summ_tokens, text_tokens, 3)
    return novelty1, novelty2, novelty3, len(summ_tokens), len(text_tokens)


def main():
    ap = ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--test-aic')
    ap.add_argument('--test-ao')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--output')
    args = ap.parse_args()
    lenghts = []
    methods = []
    res = {}

    with open(args.test_aic) as mf:
        aic_corpus = [' '.join(json.loads(e)['source']) for e in mf]
    with open(args.test_ao) as mf:
        ao_corpus = [' '.join(json.loads(e)['source']) for e in mf]

    for f in tqdm(glob.glob(args.file + '/*.hypo')):
        data = []
        with open(f) as fin:
            for line in fin:
                data.append(line.strip())
        method = f.split('/')[-1]
        text = aic_corpus if 'aic' in f else ao_corpus
        dataset = [{'summary': e1, 'text': e2} for e1, e2 in zip(data, text)]

        if args.workers > 1:
            with Pool(args.workers) as p:
                results = list(tqdm(p.imap(get_stats, dataset), total=len(dataset), unit_scale=1))
        else:
            results = [get_stats(d) for d in tqdm(dataset)]
        res[method] = np.array(results).mean(axis=0)
    
    df = pd.DataFrame(res)
    df.T.to_csv(args.output)


if __name__ == '__main__':
    main()
