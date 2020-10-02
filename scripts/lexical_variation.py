import argparse
import logging
import spacy
import json
import multiprocessing
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import join

nlp = spacy.load('en_core_sci_sm', disable=['ner', "tagger"])

def get_ngrams(orig_text):
    doc = nlp(orig_text)
    if args.lemma:
        text = [word.lemma_ for word in doc]
    else:
        text = [word.text for word in doc]

    if len(text) < args.n_gram:
        text = orig_text.split()
        return set(text), len(text)

    return set([" ".join([s.lower() for s in text[i:i+args.n_gram]]) for i in range(len(text)-args.n_gram+1)]), len(text)

def compare_docs(batches):
    auth, lauth = get_ngrams(batches['auth'])
    pr, lpr = get_ngrams(batches['pr'])

    # Tokens in pr not in auth
    intersection = pr.intersection(auth)
    union = pr.union(auth)

    if min(len(auth), len(pr)) != 0:
        containment_jaccard = len(intersection) / min(len(auth), len(pr))
    else:
        containment_jaccard = 0.

    if len(union) != 0:
        jaccard = len(intersection) / len(union)
    else:
        jaccard = 0.

    return [jaccard, containment_jaccard, lauth, lpr]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('--num_cores', default=-1, type=int)
    parser.add_argument('--lemma', default=False, action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    start = time.time()

    batches = []
    for fname in ['dev.jsonl', 'test-multitarget.jsonl']:
        papers = open(join(args.datadir, fname)).readlines()
        papers = map(lambda x: json.loads(x.strip()), papers)
        for p in papers:
            if len(p['target']) > 1:
                batches.append({
                    'auth': p['target'][0],
                    'pr': p['target'][1],
                    'id': p['paper_id']
                })

    logging.info('{} batches'.format(len(batches)))

    if args.num_cores != -1:
        num_cores = args.num_cores
    else:
        num_cores = multiprocessing.cpu_count() // 2 - 1

    logging.info('Using {} cores...'.format(num_cores))

    fig_jaccard, axs_jaccard = plt.subplots(3)
    fig_jaccard.suptitle('Jaccard Histograms')
    fig_c_jaccard, axs_c_jaccard = plt.subplots(3)
    fig_c_jaccard.suptitle('Containment Jaccard Histograms')

    for i in range(1,4):
        args.n_gram = i
        with multiprocessing.Pool(num_cores) as mp:
            data = mp.map(compare_docs, tqdm(batches))
        
        data = list(filter(lambda x: 0 if x is None else 1, data))
        jaccard = np.array([d[0] for d in data], dtype=np.float64)
        containment_jaccard = np.array([d[1] for d in data], dtype=np.float64)
        if args.n_gram == 1:
            lauth = np.array([d[2] for d in data], dtype=np.float64)
            lpr = np.array([d[3] for d in data], dtype=np.float64)
            print(f'length author')
            print(f'Mean: {np.mean(lauth)}')
            print(f'STD: {np.std(lauth)}')

            print(f'length pr')
            print(f'Mean: {np.mean(lpr)}')
            print(f'STD: {np.std(lpr)}')

        print(f'{args.n_gram}-gram jaccard')
        print(f'Mean: {np.mean(jaccard)}')
        print(f'STD: {np.std(jaccard)}')

        print(f'{args.n_gram}-gram containment jaccard')
        print(f'Mean: {np.mean(containment_jaccard)}')
        print(f'STD: {np.std(containment_jaccard)}')

        axs_jaccard[i-1].hist(jaccard, bins=100)
        # axs_jaccard.set_title(f'{args.n_gram}-gram')
        axs_c_jaccard[i-1].hist(containment_jaccard, bins=100)
        # axs_c_jaccard.set_title(f'{args.n_gram}-gram')
        fig_jaccard.savefig('jaccard.png')
        fig_c_jaccard.savefig('containment_jaccard.png')

        # print(f'{args.n_gram}-gram percent novel')
        # print(f'Mean: {np.mean(percent_novel)}')
        # print(f'STD: {np.std(percent_novel)}')

    end = time.time()
    
    logging.info('Time to run script: {} min'.format((end - start) / 60))

