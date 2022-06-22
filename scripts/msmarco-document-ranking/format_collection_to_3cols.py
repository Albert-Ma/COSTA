import os
import string
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('corpus_file')
parser.add_argument('out_file')
parser.add_argument('data_dir')

args = parser.parse_args()

did2index = {}

nltk_tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
url_commom_words = ['http', 'com', 'cn', 'org', 'https', 'www', 'html', 'htm', 'asp', 'js']

def filter_url(url):
    string_tokens = nltk_tokenizer.tokenize(url)
    return [tok for tok in string_tokens if tok not in url_commom_words]

with open(args.corpus_file) as fin, open(args.out_file, 'w') as fout:
    for index, line in enumerate(tqdm(fin, desc="Loading Dataset", unit=" lines")):
        if len(line) == 0 or len(line.split('\t'))!=4:
            print(line)
            continue
        data = line.split('\t')

        if len(data) < 3:
            print(line)
            continue

        doc_id = data[0].strip()
        assert doc_id not in did2index
        did2index[doc_id] = str(index)

        doc_url = data[1].strip()
        url_toks = filter_url(doc_url)

        doc_title = data[2].strip()
        if len(doc_title) == 0 or  doc_title in string.punctuation or doc_title == '.' or not doc_title:
            # print('title', data[2])
            doc_title = '-'
        assert len(doc_title) > 0
        # doc_text = doc_url + ' ' + data[3].strip() if len(data)>3 else ''
        # doc_text = ' '.join(url_toks) + data[3].strip() if len(data)>3 else '-'
        doc_text = doc_url + '[SEP]' + data[3].strip()

        assert len(doc_text) > 0
        fout.write("{}\t{}\t{}\n".format(did2index[doc_id], doc_title, doc_text))

did2index_file = os.path.join(args.data_dir, 'did2index.npy')
np.save(did2index_file, did2index)
