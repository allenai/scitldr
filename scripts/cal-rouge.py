""" script to calculate rouge scores on individual predictions

example command:

python scripts/calc_rouge.py \
predictions.txt \
gold/test-multitarget.jsonl \
--output results.csv --workers 30
"""

from argparse import ArgumentParser
import json
import tempfile
from files2rouge import files2rouge
import os
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd
import glob
import pathlib
import re
from multiprocessing.pool import Pool
from pathlib import Path


def filter_rouge(output_string):
    reg = "ROUGE-(1|2|L) Average_(R|P|F): (\d.\d+)"
    lines = output_string.split('\n')
    _j = {}
    for l in lines:
        if re.search(reg, l):
            match = re.search(reg, l)
            r_type = f'rouge-{match.group(1)}'.lower() # {1,2,L}
            m_type = match.group(2).lower() # {R, P, F}
            value = eval(match.group(3))
            # import ipdb; ipdb.set_trace()
            if r_type not in _j:
                _j[r_type] = {}
            elif 'f' == m_type:
                _j[r_type] = value
    # res = {}
    # for k, v in rouge.items():
    #     if 'f_score' in k:
    #         res[k] = v
    return _j

def get_rouge(args):
    return _get_rouge(args['pred'], args['data'])

def _get_rouge(pred, data):
    """ given a prediction (pred) and a data object, calculate rouge scores 
    pred: str
    data: {'target': str, '': ...}

    returns (author rouge score, multi-target rouge score (by max), multi-target rouge score (by mean))
    """
    rouge_author_score = {}
    rouge_multi_mean = defaultdict(list)
    with tempfile.TemporaryDirectory() as td:
        cand_file = os.path.join(td, 'cand')
        max_curr_rouge = -1000
        author_rouge_f1 = 0
        curr_rouge = {}
        with open(cand_file, 'w') as fh:
            fh.write(pred.strip())
        if not isinstance(data['target'], list):  # handle single target
            data['target'] = [data['target']]    
        log_file = os.path.join(td, 'rouge.log')
        for i, gold_tldr in enumerate(data['target']):
            gold_file = os.path.join(td, 'gold')
            with open(gold_file, 'w') as fh:
                fh.write(gold_tldr.strip())
            files2rouge.run(cand_file, gold_file, ignore_empty=True, saveto=log_file)
            rouge_score = Path(log_file).read_text()
            rouge_score = filter_rouge(rouge_score)
            if max_curr_rouge < rouge_score['rouge-1']:
                curr_rouge = rouge_score
                max_curr_rouge = rouge_score['rouge-1']
            if i == 0:
                rouge_author_score = rouge_score
                author_rouge_f1 = rouge_score['rouge-1']
            for k, v in rouge_score.items():
                rouge_multi_mean[k].append(v)
    for k, v in rouge_multi_mean.items():
        rouge_multi_mean[k] = sum(v) / len(v)
    rouge_multi_max = curr_rouge
    return rouge_author_score, rouge_multi_max, rouge_multi_mean


def process(gold_file, candidate_file, method_name, args):
    with open(gold_file) as fin:
        all_data = [json.loads(line) for line in fin]
    with open(candidate_file) as fin:        
        all_preds = [line.strip() for line in fin]
    all_rouges = []
    all_rouges_author = []
    all_rouges_pr = []
    count_diff_rouge = 0

    data = [{'pred': pred, 'data': data} for pred, data in zip(all_preds, all_data)]
    if args.workers > 1:
        with Pool(args.workers) as p:
            results = list(tqdm(p.imap(get_rouge, data), total=len(data), unit_scale=1))
    else:
        results = [get_rouge(d) for d in tqdm(data)]
    
    count_diff = 0
    for e1, e2, e3 in results:
        if e1['rouge-1'] != e2['rouge-1']:
            count_diff += 1
    print(count_diff, len(results))

    df_author = pd.DataFrame([e[0] for e in results], columns=['rouge-1', 'rouge-2', 'rouge-l'])
    df_multi_max = pd.DataFrame([e[1] for e in results], columns=['rouge-1', 'rouge-2', 'rouge-l'])
    df_multi_mean = pd.DataFrame([e[2] for e in results], columns=['rouge-1', 'rouge-2', 'rouge-l'])

    columns = ['rouge-1', 'rouge-2', 'rouge-l']
    all_dfs = pd.DataFrame()
    for metric_type, df in zip(['author', 'multi-max', 'multi-mean'], (df_author, df_multi_max, df_multi_mean)):
        new_columns = [f'R1||{metric_type}||{method_name}', f'R2||{metric_type}||{method_name}', f'RL||{metric_type}||{method_name}']
        all_dfs = pd.concat([all_dfs, df], axis=1)
        all_dfs = all_dfs.rename(columns={e1: e2 for e1, e2 in zip(columns, new_columns)})

    return all_dfs



def parse_args():
    ap = ArgumentParser()
    ap.add_argument('candidate')
    ap.add_argument('gold')
    ap.add_argument('--output')
    ap.add_argument('--run-over-dir', default=False, action='store_true', help='run over all the methods in the directory')
    ap.add_argument('--workers', default=1, type=int)
    ap.add_argument('--regex', default=None)
    args = ap.parse_args()
    return args

def main():
    args = parse_args()
    if args.run_over_dir:
        files = glob.glob(args.candidate + '/**/*.hypo', recursive=True)
        all_dfs = pd.DataFrame()
        for f in tqdm(files):
            if args.regex:
                if args.regex not in f:
                    continue
            method_name = f.split('/')[-1]
            df = process(args.gold, f, method_name, args)
            all_dfs = pd.concat([all_dfs, df], axis=1)
    else:        
        all_dfs = process(args.gold, args.candidate, args.candidate.split('/')[-1], args)
        print(all_dfs.mean())
    if args.output:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        all_dfs.to_csv(args.output)


if __name__ == '__main__':
    main()
