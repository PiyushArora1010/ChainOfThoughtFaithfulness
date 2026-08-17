import os
import torch
import asyncio
from openai import AsyncOpenAI
import numpy as np

from engine.base import BaseEngine
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from module.utils import print0, EMA, AverageMeter

class TrainSimulationEngine(BaseEngine):
    def __init__(self, args):
        super().__init__(args)
        self.output_dir = os.path.join(
            args.output_dir,
            "results",
            "training_simulation",
            self.dataset_tag,
            args.run_name
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.alpha = 0.99
        self.implied_client = None

        self.AS_EMA = EMA(value=self.answer_switch_ema, alpha=self.alpha)
        self.AS_EMA_BASE = AverageMeter(value=self.answer_switch_ratio, count=1, patience=100)
        print0(f"Answer switch EMA initialized to: {self.AS_EMA()}", local_rank=self.local_rank)
        print0(f"Answer switch ratio EMA initialized to: {self.AS_EMA_BASE()}", local_rank=self.local_rank)
        
        self.faithfulness_switch = EMA(value=None, alpha=self.alpha)
        self.faithfulness_non_switch = EMA(value=None, alpha=self.alpha)

        self._get_implied_client()
        self._get_base_client()

    def _get_implied_client(self):
        self.implied_client = AsyncOpenAI(
            api_key="EMPTY",  # vLLM ignores this
            base_url=self.implied_model_url,
        )
    
    def _get_base_client(self):
        self.base_client = AsyncOpenAI(
            api_key="EMPTY",  # vLLM ignores this
            base_url=self.base_model_url,
        )

    async def _get_client_responses(self, prompts):
        async def _call(prompt):
            resp = await self.implied_client.chat.completions.create(
                model=self.implied_model_tag,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.implied_model_max_tokens,
                temperature=0,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": self.implied_model_thinking
                    }
                },
                n=1,
            )
            return resp.choices[0].message.content

        tasks = [_call(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    async def _get_base_client_responses(self, prompts):
        async def _call(prompt):
            resp = await self.base_client.chat.completions.create(
                model=self.model_tag,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_seq_length,
                temperature=0,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": self.model_thinking,
                    }
                },
                n=1,
            )
            return resp.choices[0].message.content

        tasks = [_call(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    # MAIN MODEL AND TOKENIZER
    def _get_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_tag,
            trust_remote_code=True,
            fix_mistral_regex=True if "qwen" in self.model_tag.lower() else False,
        )
        tokenizer.padding_side = 'left'

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_tag,
            device_map={"": self.device},   # or "auto"
            trust_remote_code=True,
        )

        if self.lora:
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_rank,
                target_modules=self.lora_layers,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        model = model.to(self.device)

        return model, tokenizer
        
    def _prepare_datasets(self, tokenizer):
        if type(self.dataset) is dict:
            for k, v in self.dataset.items():
                v._to_hf_dataset(
                    tokenizer=tokenizer,
                    question_wrapper=self.base_prompt_answer,
                    engine=self
                )

            datasets_list = list(self.dataset.values())
            dataset = torch.utils.data.ConcatDataset(datasets_list)

            weights = []
            for d in datasets_list:
                n = len(d)
                weights.extend([1.0 / n] * n)

            self.sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )

            return dataset
        else:
            self.dataset._to_hf_dataset(
                tokenizer=tokenizer,
                question_wrapper=self.base_prompt_answer,
                engine=self
            )
            self.sampler = None
            return self.dataset
                
    def _get_model_responses(self, model, tokenizer, prompts, **kwargs):
        model_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,  
            max_length=self.max_seq_length,
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=self.max_seq_length,
                do_sample=False,
                use_cache=True,
                **kwargs
            )

        # Remove prompt tokens from generated output
        generated_ids = generated_ids[:, model_inputs["input_ids"].shape[1]:]

        responses = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        return responses

    async def _get_simulated_responses_async(
        self,
        counterfactual_full_questions,
        original_full_questions,
        original_cots,
        original_answers,
    ):
        simulated_cot_prompts = [
            self.simulated_cot_prompt.format(
                original_question=original_full_question,
                original_reasoning=original_cot,
                original_answer=original_answer,
                counterfactual_question=counterfactual_full_question,
            )
            for counterfactual_full_question, original_full_question, original_cot, original_answer in zip(
                counterfactual_full_questions, original_full_questions, original_cots, original_answers
            )
        ]

        # No asyncio.run here anymore — we're already inside a running loop
        simulated_cot_responses = await self._get_client_responses(simulated_cot_prompts)
        simulated_cot_answers = self._get_answers_from_responses(simulated_cot_responses)

        return simulated_cot_responses, simulated_cot_answers, simulated_cot_prompts

    async def _get_base_responses(self, prompts):
        base_responses = await self._get_base_client_responses(prompts)
        base_answers = self._get_answers_from_responses(base_responses)
        return base_responses, base_answers

    def _get_answers_from_responses(self, responses):
        return [self._parse_answer(response) for response in responses]

    def _get_reasoning_from_responses(self, responses):
        return [self._parse_reasoning(response) for response in responses]

    def _reward_switch(self):
        THRESH = self.AS_EMA_BASE() * self.answer_switch_factor
        if self.AS_EMA() <= THRESH:
            return (1 - self.AS_EMA() / (2 * THRESH))
        else:
            return (1 - self.AS_EMA()) / (2 * (1 - THRESH))

    def _reward(self, counterfactual_answers, simulated_cot_answers, original_answers):
        rewards = []
        meta_data = {
            "faithful_switch": 0,
            "faithful_non_switch": 0,
            "unfaithful_switch": 0,
            "unfaithful_non_switch": 0,
            "total_switch": 0,
            "total_non_switch": 0,
        }
        for counterfactual, simulated_cot, original in zip(counterfactual_answers, simulated_cot_answers, original_answers):
            if counterfactual is None or simulated_cot is None or original is None:
                rewards.append(0)
                continue

            switched = counterfactual != original
            faithful = simulated_cot == counterfactual

            reward = self._reward_switch() if switched else (1 - self._reward_switch())
            rewards.append(reward if faithful else 0)

            # Update meta_data
            if switched:
                meta_data["total_switch"] += 1
                if faithful:
                    meta_data["faithful_switch"] += 1
                else:
                    meta_data["unfaithful_switch"] += 1
            else:
                meta_data["total_non_switch"] += 1
                if faithful:
                    meta_data["faithful_non_switch"] += 1
                else:
                    meta_data["unfaithful_non_switch"] += 1

        if meta_data["total_switch"] > 0:
            self.faithfulness_switch.update(meta_data["faithful_switch"] / meta_data["total_switch"])
        if meta_data["total_non_switch"] > 0:
            self.faithfulness_non_switch.update(meta_data["faithful_non_switch"] / meta_data["total_non_switch"])
        if (meta_data["total_switch"] + meta_data["total_non_switch"]) > 0:
            self.AS_EMA.update(meta_data["total_switch"] / (meta_data["total_switch"] + meta_data["total_non_switch"]))

        return rewards
