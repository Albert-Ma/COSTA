#!/bin/bash

# copy the tokenizer file to sub-ckpt dir first 

model_dir=$1


python -m tevatron.driver.encode \
  --output_dir ${model_dir}/encoding \
  --model_name_or_path ${model_dir} \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 500 \
  --encode_in_path marco_doc_ance/bert/query/trec.query.json \
  --encoded_save_path ${model_dir}/encoding/query/trec.pt

# Step2: Search and Format the result

python -m tevatron.faiss_retriever \
  --query_reps ${model_dir}/encoding/query/trec.pt \
  --passage_reps ${model_dir}/encoding/corpus/'*.pt' \
  --depth 1000 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to ${model_dir}/encoding/trec.rank.tsv


python ../msmarco-document-ranking/score_to_trec.py ${model_dir}/encoding/trec.rank.tsv marco_doc_ance/did2index.npy


# Step3: Eval script

# trec_eval -c -m recall.100 -c -m ndcg_cut.10 qrels candids

