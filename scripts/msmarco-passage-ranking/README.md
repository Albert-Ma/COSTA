# MS-MARCO Passage Ranking
## Get Data
Run,
```
bash get_data.sh
```
This downloads the cleaned corpus, generate BM25 negatives and tokenize train/inference data using BERT tokenizer. The process could take up to tens of minutes depending on connection and hardware.

## Train Advanced Dense Retrieval Models

- We first train the pre-trained model with BM25 negatives, see [run.sh](./run.sh) for the detail training and evaluation.

- After training, we can mine hard negatives using the trained models, see [mine_hard_neg.sh](./mine_hard_neg.sh).

- We then construct the training data for hard negatives:
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

See [eval_trec_pas.sh](../../eval/eval_trec_pas.sh).