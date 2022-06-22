import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('qrel_file')
args = parser.parse_args()

with open(args.qrel_file) as f:
    lines = f.readlines()

all_qrels = set()

for line in lines:
    if len(line.strip()) == 0:
        continue
    qid, _, did, label = line.strip().split()
    label = float(label)
    if label < 1:
        continue
    all_qrels.add((qid, did))

with open(args.qrel_file + '.tv', 'w') as f:
    for qid, did in all_qrels:
        f.write("{} {}\n".format(qid,did))
