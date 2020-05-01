import torch
from fairseq.models.bart import BARTModel
import argparse
from pprint import pprint
from tqdm import tqdm
import os
from os.path import join
import shutil
import logging
import numpy as np
import json 
import random
import string
import files2rouge
import time

def test_rouge(cand, ref, outpath=None, tmp_dir='/tmp/'):
    def random_string(stringLength=8):
        """Generate a random string of fixed length """
        letters= string.ascii_lowercase
        return ''.join(random.sample(letters,stringLength))
    tmp_path = join(tmp_dir, 'tmp'+random_string())
    os.makedirs(tmp_path)
    hyp_path = join(tmp_path, 'hyp.txt')
    ref_path = join(tmp_path, 'ref.txt')

    candidates = [line.strip().lower() for line in open(cand, encoding='utf-8')]
    references = [json.loads(line.strip())['target'] for line in open(ref, encoding='utf-8')]
    paper_ids = [json.loads(line.strip())['paper_id'] for line in open(ref, encoding='utf-8')]
    
    assert len(candidates) == len(references), f'{tmp_dir}: len cand {len(candidates)} len ref {len(references)}'

    all_scores = []
    save_scores = []

    # For each prediction
    for cand_idx, cand in enumerate(candidates):
        curr_targets = references[cand_idx]
        curr_scores = []
        hyp = open(join(tmp_path, 'hyp.txt'), 'w')
        hyp.write(cand)
        hyp.close()
        # For each target
        for tgt in curr_targets:
            tgt = tgt.lower().strip('\n')
            ref = open(join(tmp_path, 'ref.txt'), 'w')
            ref.write(tgt)
            ref.close()
            try:
                _r = files2rouge.run(ref_path, hyp_path, to_json=True)
            except Exception as e:
                print(e)
                exit(0)
            curr_scores.append(_r)
        # Take the max of curr scores
        r1 = [r['rouge-1']['f'] for r in curr_scores]
        max_idx = r1.index(max(r1))

        save_scores.append({
                        'paper_id': paper_ids[cand_idx],
                        'all_scores': curr_scores,
                        'max_idx': max_idx,
                        'prediction': cand,
                        'target': curr_targets
                            })
        all_scores.append(curr_scores[max_idx])

    # Average across all scores
    avg_scores = {"rouge-1": {
                    "f": [],
                    "p": [],
                    "r":[]
                    },
                "rouge-2": {
                    "f": [],
                    "p": [],
                    "r": []
                    },
                "rouge-l": {
                    "f": [],
                    "p": [],
                    "r": []
                    }
                }
    # Append all scores to an array, the average over array
    for score in all_scores:
        for r_type in score.keys():
            for m_type in score[r_type].keys():
                x = score[r_type][m_type]
                avg_scores[r_type][m_type].append(x)   
    for r_type in avg_scores.keys():
        for m_type in avg_scores[r_type].keys():
            x = avg_scores[r_type][m_type]
            avg_scores[r_type][m_type] = np.mean(x)

    if outpath:
        with open(outpath, 'w') as fout:
            for s in save_scores:
                fout.write(json.dumps(s) + '\n')

    shutil.rmtree(tmp_path)
    return avg_scores

def evaluate(bart, bsz, count, datadir, outdir, decoder_params,
            test_fname='test.hypo', multitarget=False, quick=False):
    if torch.cuda.is_available():
        bart.cuda()
        bart.half()
    bart.eval()
    source_fname = os.path.join(datadir, 'test.source')
    pred_fname = os.path.join(outdir, test_fname)
    with open(source_fname, encoding="utf-8") as source, open(pred_fname, 'w', encoding="utf-8") as fout:
        sline = source.readline().strip()
        # sline = f'{sline} {decoder_params["ctrl"]} .'
        slines = [sline]
        for sline in tqdm(source):
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=decoder_params['beam'], 
                                                    lenpen=decoder_params['lenpen'], 
                                                    max_len_b=decoder_params['max_len_b'],
                                                    min_len=decoder_params['min_len'],
                                                    no_repeat_ngram_size=decoder_params['no_repeat_ngram_size'])
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=decoder_params['beam'], 
                                                    lenpen=decoder_params['lenpen'], 
                                                    max_len_b=decoder_params['max_len_b'],
                                                    min_len=decoder_params['min_len'],
                                                    no_repeat_ngram_size=decoder_params['no_repeat_ngram_size'])
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis.replace('\n', ' ') + '\n')
                fout.flush()
    ref_fname = 'test.jsonl' 
    ref_fname = os.path.join(datadir, ref_fname)
    r = test_rouge(pred_fname, 
                    ref_fname, 
                    outpath=os.path.join(outdir, test_fname + '.rouge'))

    return r

def maybe_percentages(r, percentages):
    if percentages:
        for r_type in ['rouge-1', 'rouge-2', 'rouge-l']:
            for m_type in ['f', 'p', 'r']:
                x = r[r_type][m_type]
                r[r_type][m_type] = x * 100
    return r

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('checkpoint_dir')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--percentages', action='store_true', default=False, 
                        help='flag if you want to print as percentages')

    # Decoder params
    # parser.add_argument('--ctrl', default='<|TLDR|>')
    parser.add_argument('--count', default=1, type=int)
    parser.add_argument('--batch_size', '--bsz', default=32, type=int, dest='bsz')
    parser.add_argument('--test_fname', default='test.hypo')
    parser.add_argument('--beam', default=6, type=int)
    parser.add_argument('--lenpen', default=1.0, type=float)
    parser.add_argument('--max_len_b', default=30, type=int)
    parser.add_argument('--min_len', default=5, type=int)
    parser.add_argument('--no_repeat_ngram_size', default=3, type=int)
    args = parser.parse_args()

    start = time.time()
    #### Path checks
    if not os.path.exists(args.datadir):
        print(f'{args.datadir} does not exist')
        exit(0)
    if not os.path.exists(join(args.datadir, 'test.source')):
        print(f'{join(args.datadir, "test.source")} does not exist')
        exit(0)
    if (not os.path.exists(join(args.checkpoint_dir, args.checkpoint_file))):
        print(f'{join(args.checkpoint_dir, args.checkpoint_file)} does not exist')
        exit(0)

    if not args.outdir:
            args.outdir = args.checkpoint_dir
    
    os.makedirs(args.outdir, exist_ok=True)

    if args.datadir.endswith('/'):
        args.datadir = args.datadir[:-1]

    bart = BARTModel.from_pretrained(
        args.checkpoint_dir,
        checkpoint_file=args.checkpoint_file,
        data_name_or_path=args.datadir + '-bin',
        task='translation'
    )

    decoder_params ={
        # 'ctrl': args.ctrl,
        'beam': args.beam,
        'lenpen': args.lenpen,
        'max_len_b': args.max_len_b,
        'min_len': args.min_len, 
        'no_repeat_ngram_size': args.no_repeat_ngram_size
    }

    r = evaluate(bart, args.bsz, args.count, 
            args.datadir, args.outdir, 
            decoder_params, 
            test_fname=args.test_fname,
            )
    r['beam'] = args.beam
    r['lenpen'] = args.lenpen
    pprint(maybe_percentages(r, args.percentages))
    
    with open(join(args.outdir, args.test_fname + '.score'), 'w') as fout:
        fout.write(json.dumps(r, indent=4))

    end = time.time()
    print(f'Time to run script: {(end-start)} sec')
