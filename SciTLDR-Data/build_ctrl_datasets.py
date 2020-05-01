import json
import os
import argparse

load_j = lambda x: json.loads(x.strip())
to_json = lambda fname: list(map(load_j, open(fname).readlines()))

def save_lines(lines, outpath):
    with open(outpath, 'w') as fout:
        for l in lines:
            fout.write(json.dumps(l) + '\n')

def add_ctrl(lines, ctrl):
    ctrl_added = []
    for l in lines:
        l['source'] += [ctrl]
        ctrl_added.append(l)
    return ctrl_added

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('--outdir')
    args = parser.parse_args()

    FNAMES = ['train.jsonl', 'dev.jsonl', 'test.jsonl']

    if not args.outdir:
        args.outdir = os.path.join(args.datadir, 'ctrl')

    # Make ctrl test files
    os.makedirs(args.outdir, exist_ok=True)

    for fname in FNAMES:
        tldrs = to_json(os.path.join(args.datadir, fname))
        tldrs = add_ctrl(tldrs, '<|TLDR|>')
        save_lines(tldrs, os.path.join(args.outdir, fname))