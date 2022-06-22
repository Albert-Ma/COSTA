#!/bin/bash

model_dir=./model
output_dir=./output

python -m tevatron.driver.train \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_dir} \
  --save_steps 20000 \
  --fp16 \
  --save_strategy epoch \
  --train_dir ./marco_doc/bert/train-hn200_pas366 \
  --per_device_train_batch_size 8 \
  --learning_rate 5e-6 \
  --num_train_epochs 1 \
  --train_n_passages 8 \
  --q_max_len 32 \
  --p_max_len 512 \
  --dataloader_num_workers 10


#  Step1: Encoding Corpus and Queries
model_dir=./${output_dir}
mkdir -p ${model_dir}/encoding/corpus
mkdir -p ${model_dir}/encoding/query


for i in  $(seq -f "%02g" 0 9)
do
python -m tevatron.driver.encode \
  --output_dir ${model_dir}/encoding \
  --model_name_or_path ${model_dir} \
  --per_device_eval_batch_size 64 \
  --p_max_len 512 \
  --fp16 \
  --encode_in_path marco_doc/bert/corpus/split${i}.json \
  --encoded_save_path ${model_dir}/encoding/corpus/split${i}.pt
done


python -m tevatron.driver.encode \
  --output_dir ${model_dir}/encoding \
  --model_name_or_path ${model_dir} \
  --q_max_len 32 \
  --encode_is_qry \
  --fp16 \
  --per_device_eval_batch_size 100 \
  --encode_in_path marco_doc/bert/query/dev.query.json \
  --encoded_save_path ${model_dir}/encoding/query/qry.pt

# Step2: Search and Format the result

python -m tevatron.faiss_retriever \
  --query_reps ${model_dir}/encoding/query/qry.pt \
  --passage_reps ${model_dir}/encoding/corpus/'*.pt' \
  --depth 1000 \
  --batch_size 5000 \
  --save_text \
  --save_ranking_to ${model_dir}/encoding/dev.rank.tsv

python ../msmarco-document-ranking/score_to_marco.py ${model_dir}/encoding/dev.rank.tsv marco_doc_ance/did2index.npy
python ../msmarco-document-ranking/score_to_trec.py ${model_dir}/encoding/dev.rank.tsv marco_doc_ance/did2index.npy

# Step3: Eval script

python ../msmarco-document-ranking/msmarco_doc_eval_anserini.py \
    --judgments  marco_doc_ance/msmarco-docdev-qrels.tsv \
    --run ${model_dir}/encoding/dev.rank.tsv.marco 
    
./eval_msmarco_doc.sh \
  ./marco_doc_ance/msmarco-docdev-qrels.tsv \
  ${model_dir}/encoding/dev.rank.tsv.marco