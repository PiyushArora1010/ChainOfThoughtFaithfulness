#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/data/sandbar/.cache"

SAVE_DIR="./"
MODEL_TAG="Qwen/Qwen3-32B"

python template_generation.py --dataset_tag cancer --dataset_path "module/datasets/data/cancer_with_split.csv" --split train \
--model_max_tokens 1024 --output_dir "$SAVE_DIR" --model_tag $MODEL_TAG \
