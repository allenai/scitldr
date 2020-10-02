import torch
from fairseq.models.bart import BARTModel
import argparse
from tqdm import tqdm
import os
from os.path import join
import logging
import time

def generate_TLDRs(bsz, count, datadir, outdir, 
                    checkpoint_dir, checkpoint_file, test_fname,
                    beam, lenpen, max_len_b, min_len, no_repeat_ngram_size):
    bart = BARTModel.from_pretrained(
        checkpoint_dir,
        checkpoint_file=checkpoint_file,
        data_name_or_path=datadir + '-bin',
        task='translation'
    )
    if torch.cuda.is_available():
        bart.cuda()
        bart.half()
    bart.eval()
    source_fname = join(datadir, 'test.source')
    pred_fname = join(outdir, test_fname)
    with open(source_fname, encoding="utf-8") as source, open(pred_fname, 'w', encoding="utf-8") as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in tqdm(source):
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=beam, 
                                                    lenpen=lenpen, 
                                                    max_len_b=max_len_b,
                                                    min_len=min_len,
                                                    no_repeat_ngram_size=no_repeat_ngram_size)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=beam, 
                                                lenpen=lenpen, 
                                                max_len_b=max_len_b,
                                                min_len=min_len,
                                                no_repeat_ngram_size=no_repeat_ngram_size)
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

    generate_TLDRs(**vars(args))
    
    end = time.time()
    print(f'Time to run script: {(end-start)} sec')
