import multiprocessing
import argparse
import time
import logging
import json
import tqdm
from os.path import join, exists
from os import makedirs


def build_batches(args, file_name):
    with open(join(args.data_dir, file_name)) as f:
        return [(args, json.loads(line.strip())) for line in f]

def format_story(data):
    args, j = data
    j["paper_id"] = str(j["paper_id"]).replace('/','_')
    j['source'] = [s.replace('\n',' ').strip() for s in j['source']]
    story = "\n\n".join(j['source']) + '\n\n'
    summary = '@highlight\n\n' + '\n\n@highlight\n\n'.join(j["target"])

    j["paper_id"] = str(j["paper_id"]).replace('/', '')

    with open(join(args.out_dir, 'stories', f'{j["paper_id"]}.story'), 'w', encoding='utf-8') as f:
        f.write(story)
        f.write(summary)

def build_mapping(args, corpus_type, data):
    if corpus_type == 'dev':
        corpus_type = 'valid'
    with open(join(args.mapping_dir, f'mapping_{corpus_type}.txt'), 'w') as f:
        for _, j in data:
            f.write(str(j['paper_id']) + '\n')

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='/path/to/*.jsonl')
    parser.add_argument('--mapping_dir', type=str, help='Default: args.data_dir/mapping')
    parser.add_argument('--out_dir', type=str, help='Default: args.data_dir')
    parser.add_argument('--num_cores', default=-1, type=int, help='Default: half of machine CPUS')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, 
                        format='%(message)s')

    start = time.time()

    if not args.out_dir:
        args.out_dir = args.data_dir

    if not args.mapping_dir:
        args.mapping_dir = join(args.data_dir, 'mapping')

    makedirs(args.out_dir, exist_ok=True)
    makedirs(join(args.out_dir, 'stories'), exist_ok=True)
    makedirs(args.mapping_dir, exist_ok=True)

    if args.num_cores > 0:
        num_cores = args.num_cores
    else:
        num_cores = multiprocessing.cpu_count() // 2
    
    for fname in ['test', 'dev', 'train']:
        if exists(join(args.data_dir, f'{fname}.jsonl')):
            batches = build_batches(args, f'{fname}.jsonl')
            build_mapping(args, fname, batches)
            with multiprocessing.Pool(num_cores) as mp:
                mp.map(format_story, tqdm.tqdm(batches))

    end = time.time()

    logging.info(f'Times to run script: {(end-start)/60} min')
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
