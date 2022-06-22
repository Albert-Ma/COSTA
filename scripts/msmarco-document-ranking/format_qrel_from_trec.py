import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('qrel_file')
parser.add_argument('out_file')
parser.add_argument('did2index_file')
args = parser.parse_args()


did2index = np.load(args.did2index_file, allow_pickle=True).item()


with open(args.qrel_file) as fin, open(args.out_file, 'w') as fout:
    for line in tqdm(fin, desc="Loading Dataset", unit=" lines"):
        if len(line) == 0 or len(line.split())!=4:
            print(line)
            continue
        qid, _, did, label = line.strip().split()
        label = int(label)
        if label < 1 :
            print(line)
            if label > 1:
                print(line)
            continue
        fout.write("{}\t{}\t{}\t{}\n".format(qid, '0',did2index[did], str(label)))


