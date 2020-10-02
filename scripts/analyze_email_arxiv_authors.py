import numpy as np
import csv
from collections import defaultdict, Counter

INFILE = 'tldr_email_arxiv_authors/tldr_eval_correctness_arxiv_authors.tsv'

arxiv_id_to_blobs = defaultdict(list)
with open(INFILE) as f_in:
    reader = csv.DictReader(f_in, delimiter='\t')
    for row in reader:
        arxiv_id_to_blobs[row['arxiv_id']].append((row['chosen_variant'], row['score']))
#      Next, pick the two comparable values
base_best_pairs = []
for arxiv_id, blobs in arxiv_id_to_blobs.items():
    bests = []
    bases = []
    for blob in blobs:
        if 'best' in blob[0]:
            bests.append(blob)
        else:
            bases.append(blob)
    if len(bests) > 1:
        base = bases[0]
        if 'ao' in base:
            best = [b for b in bests if 'ao' in b[0]][0]
        else:
            best = [b for b in bests if 'aic' in b[0]][0]
    else:
        best = bests[0]
        if 'ao' in best:
            base = [b for b in bases if 'ao' in b[0]][0]
        else:
            base = [b for b in bases if 'aic' in b[0]][0]
    base_best_pairs.append((base[1], best[1]))
#     Ranks
num_base_better = 0
num_best_better = 0
num_ties = 0
for base, best in base_best_pairs:
    if int(base) > int(best):
        num_base_better += 1
    elif int(base) < int(best):
        num_best_better += 1
    else:
        num_ties += 1
print(f'Base better: {num_base_better}')
print(f'Best better: {num_best_better}')
print(f'Ties: {num_ties}')
#    Mean score
bases = [int(p[0]) for p in base_best_pairs]
bests = [int(p[1]) for p in base_best_pairs]
print(f'Base score: {np.mean(bases)} {np.std(bases)}')
print(f'Best score: {np.mean(bests)} {np.std(bests)}')
