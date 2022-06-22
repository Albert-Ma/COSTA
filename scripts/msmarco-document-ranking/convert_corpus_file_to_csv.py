import csv
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('in_file')
parser.add_argument('out_file')
args = parser.parse_args()


with open(args.in_file) as fin, open(args.out_file, "wt", newline='') as fp:
    writer = csv.writer(fp, delimiter="\t")
    i = 0
    for line in tqdm(fin):
        i+=1
        # if i ==100:
        #     break
        data = line.strip().split('\t')
        if len(data) != 3:
            print(line, data)
        writer.writerow(data)