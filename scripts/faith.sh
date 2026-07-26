#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/data/sandbar/.cache"

SAVE_DIR="./"
DATASET_TAG="cancer"
DATASET_PATH="module/datasets/data/cancer_with_split.csv"
MODEL_TAG="/data/sandbar/saved_models/results/training_simulation/diabetes/Qwen3_4B_diabetes_ema_consistency_gt/checkpoint-600/model/"
SAVE_TAG="Qwen3_4B_consistency_gt_og"
CF_KEY="original"

python hint_faithfulness.py --dataset_tag $DATASET_TAG --dataset_path $DATASET_PATH --template_path results/templates/$DATASET_TAG/generated_templates.jsonl  --split test --hint_cf \
--model_tag $MODEL_TAG --model_temperature 0 --model_max_tokens 1028 --save_tag $SAVE_TAG \
--task responses --output_dir "$SAVE_DIR" --counterfactual_question_key $CF_KEY \

python hint_faithfulness.py --dataset_tag $DATASET_TAG --dataset_path $DATASET_PATH --template_path results/templates/$DATASET_TAG/generated_templates.jsonl --split test --hint_cf \
--model_tag Qwen/Qwen3-32B --model_temperature 0 --model_max_tokens 2048 --save_tag $SAVE_TAG \
--task implied_responses,faithfulness --output_dir "$SAVE_DIR" --counterfactual_question_key $CF_KEY \