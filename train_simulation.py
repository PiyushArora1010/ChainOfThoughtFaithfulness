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

    same_idx = next(
        (i for i, (cf, orig) in enumerate(zip(counterfactual_answers, original_answers))
        if cf == orig),
        None,
    )

    diff_idx = next(
        (i for i, (cf, orig) in enumerate(zip(counterfactual_answers, original_answers))
        if cf != orig),
        None,
    )

    example_indices = []
    if same_idx is not None:
        example_indices.append(("SAME", same_idx))
    if diff_idx is not None:
        example_indices.append(("DIFFERENT", diff_idx))

    for label, idx in example_indices:
        table = [[
            original_full_questions[idx],
            original_cots[idx],
            original_answers[idx],
            engine.answer_switch_ema,
        ]]

        print0(
            f"\n\n****Original Example ({label})****\n",
            local_rank=engine.local_rank,
        )
        print0(
            tabulate.tabulate(
                table,
                headers=[
                    "Original Full Question",
                    "Original CoT",
                    "Original Answer",
                    "Answer Switch EMA",
                ],
                tablefmt="fancy_grid",
                maxcolwidths=[40, 60, 30, 20],
            ),
            local_rank=engine.local_rank,
        )

        table = [[
            counterfactual_full_questions[idx],
            counterfactual_answers[idx],
            simulated_reasoning[idx],
            simulated_cot_answers[idx],
            rewards[idx],
        ]]

        print0(
            f"\n\n****Counterfactual Example ({label})****\n",
            local_rank=engine.local_rank,
        )
        print0(
            tabulate.tabulate(
                table,
                headers=[
                    "Counterfactual Full Question",
                    "Counterfactual Answer",
                    "Simulated Reasoning",
                    "Simulated CoT Answer",
                    "Reward",
                ],
                tablefmt="fancy_grid",
                maxcolwidths=[40, 30, 60, 30, 10],
            ),
            local_rank=engine.local_rank,
        )

    if engine.local_rank == 0:
        faithfulness_table = wandb.Table(
            columns=[
                "Type",
                "Original Question",
                "Original CoT",
                "Original Answer",
                "Counterfactual Question",
                "Counterfactual Answer",
                "Simulated Reasoning",
                "Simulated CoT Answer",
                "Reward",
            ]
        )

        for label, idx in example_indices:
            faithfulness_table.add_data(
                label,
                original_full_questions[idx],
                original_cots[idx],
                original_answers[idx],
                counterfactual_full_questions[idx],
                counterfactual_answers[idx],
                simulated_reasoning[idx],
                simulated_cot_answers[idx],
                float(rewards[idx]),
            )

        wandb.log({"faithfulness/examples": faithfulness_table})
        wandb.log({"train/answer_switch_ema": engine.answer_switch_ema})

        if_answer_switch = sum(
            1 for i in range(len(rewards)) if counterfactual_answers[i] != original_answers[i]
        )

        if if_answer_switch > 0:
            if engine.faithfulness_switch is None:
                engine.faithfulness_switch = sum(
                    1 for i in range(len(rewards)) if (counterfactual_answers[i] != original_answers[i] and rewards[i] > 0)
                ) / if_answer_switch
            else:
                engine.faithfulness_switch = (
                    engine.alpha * engine.faithfulness_switch
                    + (1 - engine.alpha) * sum(
                        1 for i in range(len(rewards)) if (counterfactual_answers[i] != original_answers[i] and rewards[i] > 0)
                    ) / if_answer_switch
                )

        if if_answer_switch < len(rewards):
            if engine.faithfulness_non_switch is None:
                engine.faithfulness_non_switch = sum(
                    1 for i in range(len(rewards)) if (counterfactual_answers[i] == original_answers[i] and rewards[i] > 0)
                ) / (len(rewards) - if_answer_switch)
            else:
                engine.faithfulness_non_switch = (
                    engine.alpha * engine.faithfulness_non_switch
                    + (1 - engine.alpha) * sum(
                        1 for i in range(len(rewards)) if (counterfactual_answers[i] == original_answers[i] and rewards[i] > 0)
                    ) / (len(rewards) - if_answer_switch)
                )

        wandb.log({"train/faithfulness_switch": engine.faithfulness_switch})
        wandb.log({"train/faithfulness_non_switch": engine.faithfulness_non_switch})

        end = time.time()
        print0(
            f"\n\n****Total reward_faithfulness time: {end - start:.2f} seconds****\n",
            local_rank=engine.local_rank,
        )

    return rewards

def reward_correct_answer(prompts, completions, **kwargs):
    global engine, model, tokenizer

    gts = kwargs["gt"]
    original_answers = engine._get_answers_from_responses(completions)

    rewards = []    
    for i in range(len(completions)):
        if original_answers[i] == gts[i]:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards

async def reward_base_answer_async(prompts, completions, **kwargs):
    start = time.time()
    global engine, model, tokenizer

    gts = kwargs["gt"]
    original_answers = engine._get_answers_from_responses(completions)
    base_responses, base_answers = await engine._get_base_responses(prompts)

    rewards = []    
    for i in range(len(completions)):
        if base_answers[i] == original_answers[i] and original_answers[i] == gts[i]:
            rewards.append(1.0)
        elif base_answers[i] == original_answers[i]:
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    table = [[
        base_responses[0],
        base_answers[0],
        original_answers[0],
        gts[0],
        rewards[0],
    ]]
    print0("\n\n****Base Model Response and Answer****\n", local_rank=engine.local_rank)
    print0(
        tabulate.tabulate(
            table,
            headers=["Base Model Response", "Base Model Answer", "Original Answer", "Ground Truth", "Reward"],
            tablefmt="fancy_grid",
            maxcolwidths=[60, 30, 30, 30, 10],
        ),
        local_rank=engine.local_rank,
    )
    if engine.local_rank == 0:
        base_table = wandb.Table(
            columns=[
                "Base Model Response",
                "Base Model Answer",
                "Original Answer",
                "Ground Truth",
                "Reward",
            ]
        )

        base_table.add_data(
            base_responses[0],
            base_answers[0],
            original_answers[0],
            gts[0],
            float(rewards[0])
        )

        wandb.log({
            "base_model/example": base_table,
        })
    end = time.time()
    print0(f"\n\n****Total reward_base_answer time: {end -start:.2f} seconds****\n", local_rank=engine.local_rank)
    return rewards

def reward_format(prompts, completions, **kwargs):
    global tokenizer
    rewards = []
    avg_token_len = 0
    for completion in completions:
        num_tokens = len(tokenizer.encode(completion))
        avg_token_len += num_tokens
        if "<reasoning>" in completion and "</reasoning>" in completion and "<answer>" in completion and "</answer>" in completion:
            rewards.append(1.0 * (num_tokens <= 512))  # Reward only if the completion is well-formatted and within token limit
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
parser.add_argument('--max_prompt_length', type=int, default=8192)
parser.add_argument('--max_seq_length', type=int, default=8192)
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
parser.add_argument('--logging_steps', type=int, default=1)
parser.add_argument('--save_steps', type=int, default=25)
parser.add_argument('--eval_steps', type=int, default=25)

# LoRA settings
parser.add_argument('--lora', action='store_true')
parser.add_argument('--lora_rank', type=int, default=16)
parser.add_argument('--lora_layers', type=str, nargs='+', default=None)

# Data settings
parser.add_argument('--dataset_tag', type=str, default='ethics')
parser.add_argument('--dataset_path', type=str, default='data/e-SNLI')
parser.add_argument('--template_path', type=str, default=None)
parser.add_argument('--concept_cf', action='store_true')
parser.add_argument('--hint_cf', action='store_true')
parser.add_argument('--hint_cf_ratio', type=float, default=0.5, help="Ratio of hinted items to total items in the dataset")
parser.add_argument('--counterfactual_data_path', type=str, default="results/concept_outputs/bbq/Llama3.3_70B")
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--sample_size', type=int, default=None, help="Number of examples")
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
    train_dataset, val_dataset = engine._prepare_datasets(tokenizer)

    print0(f"Train dataset size: {len(train_dataset)}", local_rank=engine.local_rank)
    print0(f"Validation dataset size: {len(val_dataset)}", local_rank=engine.local_rank)

    # breakpoint()

    max_prompt_length = args.max_prompt_length
    max_seq_length = args.max_seq_length

    RUN_NAME = engine.run_name

    if engine.local_rank == 0:
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
        report_to="wandb" if engine.local_rank == 0 else "none",
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
        beta=0.0,
        loss_type="grpo",
        importance_sampling_level="sequence",
        mask_truncated_completions=True,
        reward_weights=args.reward_weights if hasattr(args, "reward_weights") else [1.0, 1.0, 1.0],
        multi_objective_aggregation="normalize_then_sum",

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
        eval_dataset=val_dataset,

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
