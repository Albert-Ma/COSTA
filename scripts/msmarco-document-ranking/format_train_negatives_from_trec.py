import os
import string
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('top100_file')
parser.add_argument('qrel_file')
parser.add_argument('data_dir')
parser.add_argument('did2index_file')
parser.add_argument('k', default=100, type=int)
args = parser.parse_args()


did2index = np.load(args.did2index_file, allow_pickle=True).item()

qrel = {}
with open(args.qrel_file) as fin:
    for line in tqdm(fin):
        qid, _ , did, label = line.strip().split('\t')
        if qid in qrel:
            qrel[qid].append(did)
        else:
            qrel[qid] = [did]

all_negs = defaultdict(list)

num = 0
with open(args.top100_file) as fin:
    for i, line in enumerate(tqdm(fin, desc="Loading Dataset", unit=" lines")):
        # if i > 5000:
            # break
        if len(line) == 0 or len(line.strip().split())!=6:
            print(line)
            continue
        qid, _, docid, rank, score, runstring = line.strip().split()
        did = did2index[docid]
        if did in qrel[qid]:
            num += 1
            continue
        all_negs[qid].append(did)
print('pos num in top100:{}'.format(num))

with open(os.path.join(args.data_dir, 'train.tuned.bm25.negatives.tsv'), 'w') as fout:
    for qid, negs in tqdm(all_negs.items()):
        dids = ','.join(negs[:args.k])
        fout.write("{}\t{}\n".format(qid, dids))
