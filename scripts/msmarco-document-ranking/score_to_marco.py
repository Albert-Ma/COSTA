import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('score_file')
parser.add_argument('did2index_file')
args = parser.parse_args()

did2index = np.load(args.did2index_file, allow_pickle=True).item()

index2did = {}
for did in tqdm(did2index, total=len(did2index), desc='load index...'):
    index = did2index[did]
    index2did[index] = did

with open(args.score_file) as f:
    lines = f.readlines()

all_scores = defaultdict(list)

for line in tqdm(lines, desc='load lines...'):
    if len(line.strip()) == 0:
        continue
    qid, did, score = line.strip().split()
    score = float(score)
    all_scores[qid].append((did, score))

qq = list(all_scores.keys())

with open(args.score_file + '.marco', 'w') as f:
    for qid in tqdm(qq):
        score_list = sorted(all_scores[qid], key=lambda x: x[1], reverse=True)
        for rank, (did, score) in enumerate(score_list):
            f.write(f'{qid}\t{index2did[did]}\t{rank+1}\n')

