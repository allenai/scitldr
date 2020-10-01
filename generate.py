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

def generate_TLDRs(bart, bsz, count, datadir, outdir, decoder_params,
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

def maybe_percentages(r, percentages):
    if percentages:
        for r_type in ['rouge-1', 'rouge-2', 'rouge-l']:
            for m_type in ['f', 'p', 'r']:
                x = r[r_type][m_type]
                r[r_type][m_type] = x * 100
    return r

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', help='Path to checkpoint directory')
    parser.add_argument('datadir', help='Path to data directory')
    parser.add_argument('outdir', help='Path to output directory')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt')
    parser.add_argument('--test_fname', default='test.hypo')
    
    # Decoder params
    parser.add_argument('--count', default=1, type=int)
    parser.add_argument('--batch_size', '--bsz', default=32, type=int, dest='bsz')
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
    
    os.makedirs(args.outdir, exist_ok=True)

    # Clean up to pass into fairseq
    if args.datadir.endswith('/'):
        args.datadir = args.datadir[:-1]

    bart = BARTModel.from_pretrained(
        args.checkpoint_dir,
        checkpoint_file=args.checkpoint_file,
        data_name_or_path=args.datadir + '-bin',
        task='translation'
    )

    decoder_params ={
        'beam': args.beam,
        'lenpen': args.lenpen,
        'max_len_b': args.max_len_b,
        'min_len': args.min_len, 
        'no_repeat_ngram_size': args.no_repeat_ngram_size
    }

    generate_TLDRs(bart, args.bsz, args.count, 
            args.datadir, args.outdir, 
            decoder_params, 
            test_fname=args.test_fname
            )
    

    end = time.time()
    print(f'Time to run script: {(end-start)} sec')
