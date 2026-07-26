#!/bin/bash

MODEL=$1
PORT=$2
DEVICES=$3

export CUDA_VISIBLE_DEVICES=$DEVICES
export HF_HOME="/data/sandbar/.cache/"

python -m vllm.entrypoints.openai.api_server \
--model $MODEL \
--max-model-len 8192 \
--tensor-parallel-size 1 \
--gpu-memory-utilization 0.90 \
--max-num-seqs 64 \
--dtype bfloat16 \
--port $PORT \