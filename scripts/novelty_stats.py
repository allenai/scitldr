"""
simple script for calculating novelty stats

Datasets used for these stats:

bigpatent dataset: https://evasharma.github.io/bigpatent/
clpubsum: https://github.com/EdCo95/scientific-paper-summarisation
arxiv: https://github.com/armancohan/long-summarization
scisummnet: https://cs.stanford.edu/~myasu/projects/scisumm_net/

The stats are run over the dev set of these datasets
"""


import gzip
import json
from multiprocessing.pool import Pool
from tqdm.auto import tqdm
import nltk
from nltk.util import ngrams
from argparse import ArgumentParser


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
    return novelty1, novelty2, novelty3

def parse_args():
    ap = ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--workers', default=1, type=int)
    ap.add_argument('--dataset-type', choices={'newsroom', 'scisummnet', 'bigpatent', 'arxiv', 'clpubsumm', 'scitldr'}, default='newsroom')
    args = ap.parse_args()
    return args

def main():
    args = parse_args()
    
    if args.dataset_type == 'newsroom':
        import gzip
        with gzip.open(args.input) as fin:
            dataset = [json.loads(e) for e in fin]
            dataset = [{'text': e['text'], 'summary': e['summary']} for e in dataset]
    elif args.dataset_type == 'scitldr':
        import glob
        import gzip
        with open(args.input) as fin:
            dataset = []
            for e in fin:
                obj = json.loads(e)
                # multitarget
                for ee in obj['target']:
                    new_obj = {}
                    new_obj['text'] = ' '.join(obj['source'])
                    new_obj['summary'] = ee
                    dataset.append(new_obj)
    elif args.dataset_type == 'arxiv':
        import glob
        import gzip
        with open(args.input) as fin:
            dataset = [json.loads(e) for e in fin]
            dataset = [{'text': ' '.join(e['article_text']), 
                        'summary': ' '.join(e['abstract_text']).replace('<S>', '').replace('</S>', '')} for e in dataset]
    elif args.dataset_type == 'clpubsumm':
        # tmp/sci_sum/dev-complete.jsonl
        dataset = []
        with open(args.input) as fin:
            for e in fin:
                obj = json.loads(e)
                new_obj = {}
                new_obj['text'] = ' '.join(obj['sentences'])
                summary = ''
                for ee in obj['abstract']:
                    summary += ' '.join(ee)
                new_obj['summary'] = summary
            dataset.append(new_obj)
    elif args.dataset_type == 'bigpatent':
        import glob
        import gzip
        files = glob.glob(args.input + '/**/*.gz', recursive=True)
        dataset = []
        for f in tqdm(files, desc='reading files'):
            for line in gzip.open(f, 'rt'):
                obj = json.loads(line)
                dataset.append({'text': obj['description'], 'summary': obj['abstract']})
    elif args.dataset_type == 'scisummnet':
        import xml.etree.ElementTree as ET
        import glob
        import os
        import pathlib
        files = glob.glob(args.input + '/**/*.xml', recursive=True)
        dataset = []
        for f in tqdm(files, desc='reading files'):
            summary_dir = str(pathlib.Path(f).parent.parent) + '/summary/'
            summary_fp = summary_dir + '/' + os.listdir(summary_dir)[0]
            with open(summary_fp) as fh: 
                summary = fh.read()
            tree = ET.parse(f)
            root = tree.getroot()
            try:
                text = ' '.join([child.text for child in root.findall('.//S')])
            except TypeError:
                print('cant process', f)
                continue
            dataset.append({'text': text, 'summary': summary})

        
    if args.workers > 1:
        with Pool(args.workers) as p:
            results = list(tqdm(p.imap(get_stats, dataset), total=len(dataset), unit_scale=1))
    else:
        results = [get_stats(d) for d in tqdm(dataset)]        
    novelty1 = [e[0] for e in results]
    novelty2 = [e[1] for e in results]
    novelty3 = [e[2] for e in results]
    nov1 = (sum(novelty1) / len(novelty1)) * 100
    nov2 = (sum(novelty2) / len(novelty2)) * 100
    nov3 = (sum(novelty3) / len(novelty3)) * 100
    print(f"1gram\t2gram\t3gram")
    print(f"{nov1:.2f}\t{nov2:.2f}\t{nov3:.2f}")


if __name__ == '__main__':
    main()
