#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5
export HF_HOME="./"

SAVE_DIR="./"
MODEL_TAG="Qwen/Qwen3-4B"
MODEL_SAVE_TAG="Qwen3_4B"

accelerate launch --mixed-precision=bf16 train_simulation.py --dataset_tag diabetes --dataset_path "module/datasets/data/diabetes_with_split.csv" --template_path results/templates/diabetes/generated_templates.jsonl  --model_tag $MODEL_TAG \
--model_temperature 1.0 --max_seq_length 512 --max_prompt_length 512 --model_batch_size 4 --implied_model_tag Qwen/Qwen3-32B \
--learning_rate 1e-5 --weight_decay 1e-3 --epochs 5 --gradient_accumulation_steps 2 --completions_per_prompt 8 --seed 0 \
--lora --lora_rank 64 --lora_layers q_proj v_proj o_proj k_proj \
--eval_steps 1000000 --save_steps 100 --hint_cf --answer_switch_ratio 0.25 \
--run_name "${MODEL_SAVE_TAG}_diabetes_ema_consistency_gt_as_0.25" --output_dir "$SAVE_DIR" \
--base_model_url "http://localhost:3317/v1" --reward_weights 1.0 1.0 0.1 \
