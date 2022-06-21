#!/usr/bin/env bash
# Setting for the new UTF-8 terminal support in Lion
export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# qrel file
qrel="$1"

# your candidate run
run="$2"

# maxMRRRank, default is 100
if [ $# -eq 3 ]
then
maxMRRRank="$3"
else
maxMRRRank=10
fi

# usage:
python ./ms_marco_doc_eval.py --run $run --judgments $qrel --maxMRRRank $maxMRRRank

# sh ./eval/eval_msmarco.sh /data/mxy/cooperative-irgan/data/init_discriminator/retrieved_result/top1000.rank.txt /data/mxy/cooperative-irgan/data/msmarco-passage/qrels.dev.small.tsv 10