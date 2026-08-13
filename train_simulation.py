import time
import random
import tabulate
import asyncio
import wandb
import argparse
from engine.train_simulation import TrainSimulationEngine

from module.utils import print0, set_seed
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer

async def reward_faithfulness_async(prompts, completions, **kwargs):
    start = time.time()
    global engine, model, tokenizer

    original_or_cf = random.random() < 0.5

    counterfactual_prompts_all = kwargs["counterfactual_prompt"] if original_or_cf else kwargs["original_prompt"]
    original_full_questions = kwargs["full_question"]
    counterfactual_full_questions = kwargs["counterfactual_full_question"] if original_or_cf else kwargs["original_full_question"]

    # Get only unique counterfactual prompts
    counterfactual_prompts = [
        counterfactual_prompts_all[i]
        for i in range(0, len(completions), engine.completions_per_prompt)
    ]
    print0(f"Unique counterfactual prompts: {len(counterfactual_prompts)}", local_rank=engine.local_rank)
    print0(f"Number of rollouts per counterfactual prompt: {engine.completions_per_prompt}", local_rank=engine.local_rank)

    # These are cheap/CPU-only, fine to do up front
    original_cots = engine._get_reasoning_from_responses(completions)
    original_answers = engine._get_answers_from_responses(completions)

    # --- Launch both independent stages concurrently ---
    counterfactual_task = asyncio.create_task(
        asyncio.to_thread(
            engine._get_model_responses, model, tokenizer, counterfactual_prompts
        )
    )
    simulated_task = asyncio.create_task(
        engine._get_simulated_responses_async(
            counterfactual_full_questions,
            original_full_questions,
            original_cots,
            original_answers,
        )
    )

    counterfactual_responses, simulated_results = await asyncio.gather(
        counterfactual_task, simulated_task
    )

    simulated_cot_responses, simulated_cot_answers, simulated_cot_prompts = simulated_results
    simulated_reasoning = engine._get_reasoning_from_responses(simulated_cot_responses)

    counterfactual_answers = engine._get_answers_from_responses(counterfactual_responses)
    counterfactual_answers = [
        ans for ans in counterfactual_answers for _ in range(engine.completions_per_prompt)
    ]  # repeat answers per completion

    # Faithfulness reward computation
    rewards = engine._reward(counterfactual_answers, simulated_cot_answers, original_answers)
    end = time.time()

    if engine.local_rank == 0 and engine.wandb:
        wandb.log({"train/faithfulness_switch": engine.faithfulness_switch()})
        wandb.log({"train/faithfulness_non_switch": engine.faithfulness_non_switch()})
        wandb.log({"train/answer_switch": engine.AS_EMA()})
        wandb.log({"train/answer_switch_ratio": engine.AS_EMA_BASE()})
        
    print0(f"Original Answers: {original_answers}", local_rank=engine.local_rank)
    print0(f"Counterfactual Answers: {counterfactual_answers}", local_rank=engine.local_rank)
    print0(f"Simulated CoT Answers: {simulated_cot_answers}", local_rank=engine.local_rank)
    print0(f"Faithfulness Rewards: {[f'{r:.2f}' for r in rewards]}", local_rank=engine.local_rank)
    print0(
        f"\n\n****Total reward_faithfulness time: {end - start:.2f} seconds****\n",
        local_rank=engine.local_rank,
    )

    return rewards

async def reward_base_answer_async(prompts, completions, **kwargs):
    start = time.time()
    global engine, model, tokenizer

    gts = kwargs["gt"]
    original_answers = engine._get_answers_from_responses(completions)

    unique_prompts = [
        prompts[i] for i in range(0, len(completions), engine.completions_per_prompt)
    ]
    counterfactual_unique_prompts = [
        kwargs["counterfactual_prompt"][i]
        for i in range(0, len(completions), engine.completions_per_prompt)
    ]

    _, base_answers_unique = await engine._get_base_responses(
        unique_prompts + counterfactual_unique_prompts
    )
    base_answers_unique, counterfactual_base_answers_unique = base_answers_unique[:len(unique_prompts)], base_answers_unique[len(unique_prompts):]

    base_answers = [a for a in base_answers_unique for _ in range(engine.completions_per_prompt)]
    counterfactual_base_answers = [a for a in counterfactual_base_answers_unique for _ in range(engine.completions_per_prompt)]

    rewards = []    
    meta_data = {
        "switch": 0,
        "non_switch": 0,
    }
    for i in range(len(completions)):
        if base_answers[i] == None or original_answers[i] == None:
            rewards.append(0)
            continue
        elif base_answers[i] == original_answers[i]:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

        if base_answers[i] != counterfactual_base_answers[i]:
            meta_data["switch"] += 1
        else:
            meta_data["non_switch"] += 1

    print0(f"Base Model Answers: {base_answers}", local_rank=engine.local_rank)
    print0(f"Counterfactual Base Model Answers: {counterfactual_base_answers}", local_rank=engine.local_rank)
    print0(f"Original Answers: {original_answers}", local_rank=engine.local_rank)
    print0(f"Ground Truths: {gts}", local_rank=engine.local_rank)
    print0(f"Rewards: {[f'{r:.2f}' for r in rewards]}", local_rank=engine.local_rank)

    if meta_data["switch"] + meta_data["non_switch"] > 0:
        engine.AS_EMA_BASE.update(meta_data["switch"] / (meta_data["switch"] + meta_data["non_switch"]))

    end = time.time()
    print0(f"\n\n****Total reward_base_answer time: {end-start:.2f} seconds****\n", local_rank=engine.local_rank)
    return rewards

def reward_format(prompts, completions, **kwargs):
    global tokenizer, engine
    rewards = []
    avg_token_len = 0
    for completion in completions:
        num_tokens = len(tokenizer.encode(completion))
        avg_token_len += num_tokens
        if "<reasoning>" in completion and "</reasoning>" in completion and "<answer>" in completion and "</answer>" in completion:
            rewards.append(1.0 * (num_tokens <= engine.max_seq_length))  # Reward only if the completion is well-formatted and within token limit
        else:
            rewards.append(0.0)
    print0(f"\n\n****Average token length of completions: {avg_token_len / len(completions):.2f} tokens****\n", local_rank=engine.local_rank)
    return rewards

parser = argparse.ArgumentParser()

# Implied concepts settings
parser.add_argument('--implied_model_thinking', action='store_true')
parser.add_argument('--implied_model_tag', type=str, default='Qwen/Qwen3-4B')
parser.add_argument('--implied_model_url', type=str, default="http://localhost:3316/v1")
parser.add_argument('--implied_model_max_tokens', type=int, default=2048)

# Base model settings
parser.add_argument('--base_model_url', type=str, default="http://localhost:3317/v1")

# Model settings
parser.add_argument('--model_tag', type=str, default='Qwen/Qwen3-4B')
parser.add_argument('--max_prompt_length', type=int, default=512)
parser.add_argument('--max_seq_length', type=int, default=512)
parser.add_argument('--model_temperature', type=float, default=1)
parser.add_argument('--model_batch_size', type=int, default=8)
parser.add_argument('--model_thinking', action='store_true')
parser.add_argument('--use_vllm', action='store_true', help="Use vLLM for training (if supported)")

# Training settings
parser.add_argument('--learning_rate', type=float, default=5e-6)
parser.add_argument('--weight_decay', type=float, default=1e-3) 
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--completions_per_prompt', type=int, default=6)
parser.add_argument('--reward_weights', type=float, nargs='+', default=[1.0, 1.0, 1.0], help="Weights for the reward functions: [base_answer, faithfulness, format]")
parser.add_argument('--answer_switch_ratio', type=float, default=0.5, help="Ratio of answer switches to total examples in the dataset")
parser.add_argument('--answer_switch_ema', type=float, default=None, help="Initial value for the exponential moving average of answer switches")

# Save and Logging settings
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='deleteme')
parser.add_argument('--wandb', action='store_true', help="Log to Weights & Biases")
parser.add_argument('--logging_steps', type=int, default=1)
parser.add_argument('--save_steps', type=int, default=25)
parser.add_argument('--eval_steps', type=int, default=25)

# LoRA settings
parser.add_argument('--lora', action='store_true')
parser.add_argument('--lora_rank', type=int, default=16)
parser.add_argument('--lora_layers', type=str, nargs='+', default=None)

# Data settings
parser.add_argument('--dataset_tag', type=str, default='ethics')
parser.add_argument('--hint_cf', action='store_true')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="Path to a checkpoint to resume training from")

args = parser.parse_args()
set_seed(args.seed)

if __name__ == '__main__':

    engine = TrainSimulationEngine(args)
    
    # Prepare model
    print0("Preparing model and datasets...", local_rank=engine.local_rank)
    model, tokenizer = engine._get_model_and_tokenizer()

    # Prepare datasets
    train_dataset = engine._prepare_datasets(tokenizer)
    print0(f"Train dataset size: {len(train_dataset)}", local_rank=engine.local_rank)

    max_prompt_length = args.max_prompt_length
    max_seq_length = args.max_seq_length

    RUN_NAME = engine.run_name

    if engine.local_rank == 0 and engine.wandb:
        wandb.init(
            project="Faithfulness ISO",
            name=RUN_NAME,  
            config=vars(args), 
        )

    if args.use_vllm:
        vllm_args = {
            "vllm_sampling_params" :  SamplingParams(
                seed = args.seed,
                temperature=args.model_temperature,
                max_tokens=max_seq_length,
            )
        }
    else:
        vllm_args = {}

    print0(f"Reward weights: {args.reward_weights}", local_rank=engine.local_rank)

    training_args = GRPOConfig(
        # Training
        learning_rate=args.learning_rate,
        lr_scheduler_type = "cosine_with_min_lr",
        lr_scheduler_kwargs = {"min_lr": args.learning_rate * 1e-2},
        warmup_ratio=0,
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,

        # Batching
        per_device_train_batch_size=args.model_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.model_batch_size,
        eval_accumulation_steps=args.gradient_accumulation_steps,

        # Logging / checkpointing
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        report_to="wandb" if (args.wandb and engine.local_rank == 0) else "none",
        run_name=RUN_NAME,
        output_dir=engine.output_dir,

        # Training length
        num_train_epochs=args.epochs,

        # Generation
        num_generations=args.completions_per_prompt,
        max_completion_length=max_seq_length,
        temperature=args.model_temperature,
        # use_transformers_continuous_batching=True,
        # transformers_continuous_batching_config={
        #     "use_cuda_graph": False,
        #     "max_memory_percent": 0.4,  # leave headroom for the backward pass
        # },

        # GRPO-specific
        beta=0.01,
        loss_type="luspo",
        epsilon=2e-3, # section 5.1 of the paper
        epsilon_high=2.5e-3, # section 5.1 of the paper
        importance_sampling_level="sequence",
        mask_truncated_completions=True,
        reward_weights=args.reward_weights if hasattr(args, "reward_weights") else [1.0, 1.0, 0.1],
        multi_objective_aggregation="normalize_then_sum",

        # entropy_coef=0.02,
        # use_adaptive_entropy=True,
        # entropy_target=0.35,        # calibrated near your actual step-0 entropy (~0.45), not 5.0
        # entropy_coef_delta=0.001,   # slower ramp given tight epsilon clip
        # entropy_coef_min=0.0,
        # entropy_coef_max=0.15,      # prevent entropy term from dominating your bounded [0,1.3] reward

        gradient_checkpointing = False,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,

        # vLLM
        use_vllm=args.use_vllm,
        **vllm_args,

        # Seeds
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,

        train_dataset=train_dataset,

        reward_funcs=[
            reward_base_answer_async,
            reward_faithfulness_async,
            reward_format,
        ],
    )
    print0("Starting training...", local_rank=engine.local_rank)
    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint  # Set to a checkpoint path to resume training from a saved state
    )
