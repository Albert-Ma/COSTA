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
Train a BERT(`bert-base-uncased`) with mixed precision.
```
python -m tevatron.driver.train \
  --output_dir ./retriever_model \
  --model_name_or_path bert-base-uncased \
  --save_steps 20000 \
  --train_dir ./marco/bert/train \
  --fp16 \
  --per_device_train_batch_size 8 \
  --learning_rate 5e-6 \
  --num_train_epochs 2 \
  --dataloader_num_workers 2
```

## Encode the Corpus and Query
```
mkdir encoding
for i in $(seq -f "%02g" 0 9)
do
python -m tevatron.driver.encode \
  --output_dir ./retriever_model \
  --model_name_or_path ./retriever_model \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/bert/corpus/split${i}.json \
  --encoded_save_path encoding/split${i}.pt
done


python -m tevatron.driver.encode \
  --output_dir ./retriever_model \
  --model_name_or_path ./retriever_model \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/bert/query/dev.query.json \
  --encoded_save_path encoding/qry.pt
```

### Search the Corpus
```
mkdir -p ranking/intermediate

for i in $(seq -f "%02g" 0 9)
do
python -m tevatron.faiss_retriever \
  --query_reps encoding/qry.pt \
  --passage_reps encoding/split${i}.pt \
  --depth 10 \
  --save_ranking_to ranking/intermediate/split${i}
done

python -m tevatron.faiss_retriever.reducer \
  --score_dir ranking/intermediate \
  --query encoding/qry.pt \
  --save_ranking_to ranking/rank.txt

```
Finally format the retrieval result,
```
python score_to_marco.py ranking/rank.txt
```
