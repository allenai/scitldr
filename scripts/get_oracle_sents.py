import rouge
import argparse
import json
from tqdm import tqdm

def build_batches(datapath):
    batches = []
    with open(datapath) as f:
        lines = f.readlines()
        lines = map(lambda x: json.loads(x.strip()), lines)
        for _j in lines:
            candidates, references, paper_ids = [], [], []
            candidates = _j['source']
            if type(_j['target']) == list:
                references = _j['target']
            else:
                references = [j['target']]
            paper_id = _j['paper_id']
            batches.append({
                'candidates': candidates,
                'references': references,
                'paper_ids': paper_ids
            })
    return batches

def get_oracle_single_paper(batch):
    candidates = batch['candidates']
    references = batch['references']

    evaluator = rouge.Rouge()
    max_r1 = 0.
    max_score = None
    max_sent = ''
    def check_length(c):
        l = len(c)
        if l < 5 or l > 2500:
            return 0
        return 1
    # sentences with too long or too short of characters will break the script
    candidates = [c.strip() for c in candidates if check_length(c)]
    for tgt in references:
        ref = [tgt]*len(candidates)
        scores = evaluator.get_scores(candidates, ref)
        r1 = [s['rouge-1']['f'] for s in scores]
        max_idx = r1.index(max(r1))
        if max_r1 < r1[max_idx]:
            max_r1 = r1[max_idx]
            max_score = scores[max_idx]
            max_sent = candidates[max_idx]

    return max_sent.replace('\n', '').strip()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath')
    parser.add_argument('outpath')
    args = parser.parse_args()

    batches = build_batches(args.datapath, title_path=args.title_path)
    results = [get_oracle_single_paper(d) for d in tqdm(batches)]

    with open(args.outpath, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(results))
