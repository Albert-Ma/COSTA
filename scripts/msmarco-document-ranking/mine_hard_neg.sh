#!/bin/bash


model_dir=./model_dir

mkdir -p ${model_dir}/encoding/corpus
mkdir -p ${model_dir}/encoding/query

#  Step 1: Encode corpus
for i in $(seq -f "%02g" 0 9)
do
python -m tevatron.driver.encode \
  --output_dir ${model_dir}/encoding \
  --model_name_or_path ${model_dir} \
  --per_device_eval_batch_size 64 \
  --encode_in_path marco_doc/bert/corpus/split${i}.json \
  --encoded_save_path ${model_dir}/encoding/corpus/split${i}.pt
done



# Step 2: encode training queries
python -m tevatron.driver.encode \
  --output_dir ${model_dir}/encoding \
  --model_name_or_path ${model_dir} \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco_doc/bert/query/train.query.json \
  --encoded_save_path ${model_dir}/encoding/query/train.pt


# Step 3: search hard negs for training queries

# (1) directly search
python -m tevatron.faiss_retriever \
    --query_reps ${model_dir}/encoding/query/train.pt \
    --passage_reps ${model_dir}/encoding/corpus/'*.pt' \
    --batch_size 5000 \
    --depth 200 \
    --save_text \
    --save_ranking_to ${model_dir}/encoding/train.rank.tsv


# (2) retrieval on each split and then combine them
mkdir -p ${model_dir}/encoding/doctrain_split_rank
for s in split00 split01 split02 split03 split04 split05 split06 split07 split08 split09 
do
python -m tevatron.faiss_retriever \
  --query_reps ${model_dir}/encoding/query/train.pt \
  --passage_reps ${model_dir}/encoding/corpus/${s}.pt \
  --depth 200 \
  --save_ranking_to ${model_dir}/encoding/doctrain_split_rank/${s}
done

python -m tevatron.faiss_retriever.reducer \
  --score_dir ${model_dir}/encoding/doctrain_split_rank/ \
  --query ${model_dir}/encoding/query/train.pt \
  --save_ranking_to ${model_dir}/encoding/train.rank.tsv

python ../msmarco-document-ranking/score_to_marco.py ${model_dir}/encoding/train.rank.tsv marco_doc_ance/did2index.npy
