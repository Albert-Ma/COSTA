#!/bin/bash

model_dir=./model
output_dir=./output

python -m tevatron.driver.train \
  --output_dir ./${output_dir} \
  --model_name_or_path ${model_dir} \
  --save_steps 20000 \
  --save_strategy epoch \
  --fp16 \
  --train_dir ./marco/bert/train \
  --per_device_train_batch_size 8 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --dataloader_num_workers 10


#  Step1: Encoding Corpus and Queries
mkdir -p ${model_dir}/encoding/corpus
mkdir -p ${model_dir}/encoding/query


for i in $(seq -f "%02g" 0 9)
do
python -m tevatron.driver.encode \
  --output_dir ${model_dir}/encoding \
  --model_name_or_path ${model_dir} \
  --per_device_eval_batch_size 4000 \
  --fp16 \
  --encode_in_path marco/bert/corpus/split${i}.json \
  --encoded_save_path ${model_dir}/encoding/corpus/split${i}.pt
done


python -m tevatron.driver.encode \
  --output_dir ${model_dir}/encoding \
  --model_name_or_path ${model_dir} \
  --q_max_len 32 \
  --encode_is_qry \
  --fp16 \
  --per_device_eval_batch_size 100 \
  --encode_in_path marco/bert/query/dev.query.json \
  --encoded_save_path ${model_dir}/encoding/query/qry.pt

# Step2: Search and Format the result

python -m tevatron.faiss_retriever \
  --query_reps ${model_dir}/encoding/query/qry.pt \
  --passage_reps ${model_dir}/encoding/corpus/'*.pt' \
  --depth 1000 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to ${model_dir}/encoding/dev.rank.tsv

python ../msmarco-passage-ranking/score_to_marco.py ${model_dir}/encoding/dev.rank.tsv
python ../msmarco-passage-ranking/score_to_trec.py ${model_dir}/encoding/dev.rank.tsv

# Step3: Eval script
./eval_msmarco_passage.sh \
  ./marco/qrels.dev.tsv \
  ${model_dir}/encoding/dev.rank.tsv.marco