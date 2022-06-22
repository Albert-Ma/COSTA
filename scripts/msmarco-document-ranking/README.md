# MS-MARCO Document Ranking
## Preprocess Data
- preprocess document collection
```
python ../format_collection_to_3cols.py msmarco-docs.tsv corpus.tsv
```
- preprocess train top-k, remeber to remove the positive document from top-k
```
python format_train_negatives_from_trec.py
```
- preprocess qrel to trec format
```
python format_qrel_from_trec.py
```

- build train data from corpus.tsv and train.negatives.tsv

```
python ../msmarco-document-ranking/build_train.py \
  --tokenizer_name bert-base-uncased \
  --negative_file marco_doc/train.negatives.tsv \
  --qrels marco_doc/qrels.train.tsv  \
  --queries marco_doc/msmarco-doctrain-queries.tsv  \
  --collection marco_doc/corpus.tsv \
  --save_to marco_doc/bert/hd_1st \
  --truncate 512 \
  --mp_chunk_size 10000
```

- tokenize query and corpus
```
python ../tokenize_queries.py --tokenizer_name bert-base-uncased --query_file dev.query.txt --save_to $bert/query/dev.query.json
python ../tokenize_passages.py --tokenizer_name bert-base-uncased --file corpus.tsv --save_to bert/corpus
```

Note that for MS MARCO document retrieval datasets, we map the original doc_id (e.g., Dxxxx) to interger doc_id (e.g., 1) and save the mapping to `did2index.npy`, see [format_collection_to_3cols.py](./format_collection_to_3cols.py).
The evaluation for document retrieval will use this `did2index.npy`,
```
python ../msmarco-document-ranking/score_to_marco.py ${model_dir}/encoding/dev.rank.tsv marco_doc/did2index.npy
python ../msmarco-document-ranking/score_to_trec.py ${model_dir}/encoding/dev.rank.tsv marco_doc/did2index.npy
```

## Train Advanced Dense Retrieval Models

**Note that we use the passage retrieval model as the starting point to train the MS MARCO document ranking task.**


- We first train the pre-trained model with hard negatives mined by passage retrieval model, see [mine_hard_neg.sh](./mine_hard_neg.sh) to mine hard negatives

- run the following command to construct training data for the 1st iteration:
```
python ../msmarco-passage-ranking/build_train_hn.py  \
    --tokenizer_name bert-base-uncased \
    --hn_file model_dir/encoding/train.rank.tsv \
    --qrels marco-pas/qrels.train.tsv   \
    --queries marco-pas/train.query.txt \
    --collection marco-pas/corpus.tsv \
    --save_to marco-pas/bert/train-hn
```
- and run [run.sh](./run.sh) for the 1st iteration training and evaluation.

- After the 1st round training, we can mine hard negatives for the 2nd iteration using the trained models, see [mine_hard_neg.sh](./mine_hard_neg.sh).

- We then construct the training data for hard negatives of the 2nd iteration:
```
python ../msmarco-passage-ranking/build_train_hn.py  \
    --tokenizer_name bert-base-uncased \
    --hn_file model_dir/encoding/train.rank.tsv \
    --qrels marco-pas/qrels.train.tsv   \
    --queries marco-pas/train.query.txt \
    --collection marco-pas/corpus.tsv \
    --save_to marco-pas/bert/train-hn
```

- Train the advanced dense retrieval mdoel with hard negatievs, see [train_hn.sh](./train_hn.sh).

### Training

Some details can be found [here](https://github.com/texttron/tevatron/tree/fef1f846949cada591e444b30871d9319b30a7b9).


## Evaluation on TREC datasets

See [eval_trec_doc.sh](../../eval/eval_trec_doc.sh).