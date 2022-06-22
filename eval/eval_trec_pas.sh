#!/bin/bash

# copy the tokenizer file to sub-ckpt dir first 

model_dir=$1


python -m tevatron.driver.encode \
  --output_dir ${model_dir}/encoding_w_fp16 \
  --model_name_or_path ${model_dir} \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 50 \
  --encode_in_path marco/bert/query/trec.query.json \
  --encoded_save_path ${model_dir}/encoding_w_fp16/query/trec.pt

# Step2: Search and Format the result

python -m tevatron.faiss_retriever \
  --query_reps ${model_dir}/encoding_w_fp16/query/trec.pt \
  --passage_reps ${model_dir}/encoding_w_fp16/corpus/'*.pt' \
  --depth 1000 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to ${model_dir}/encoding_w_fp16/trec.rank.tsv

python ../msmarco-passage-ranking/score_to_trec.py ${model_dir}/encoding_w_fp16/trec.rank.tsv
python ../msmarco-passage-ranking/score_to_marco.py ${model_dir}/encoding_w_fp16/trec.rank.tsv

# Step3: Eval script

# trec_eval -m all_trec  -c -l 2 qrels candids
