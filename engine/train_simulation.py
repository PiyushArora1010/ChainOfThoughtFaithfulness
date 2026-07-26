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

        self.implied_client = None
        print(f"Answer switch EMA initialized to: {self.answer_switch_ema}")
        self.alpha = 0.99
        self.faithfulness_switch = None
        self.faithfulness_non_switch = None
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
                lora_alpha=self.lora_rank * 2,
                target_modules=self.lora_layers,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        model = model.to(self.device)

        return model, tokenizer
    
    def _prepare_datasets(self, tokenizer):
        self.dataset._to_hf_dataset(
            tokenizer=tokenizer,
            question_wrapper=self.base_prompt_answer,
            engine=self
        )

        train_size = len(self.dataset) - 50
        val_size = 50
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        return train_dataset, val_dataset
                
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

    def _reward(self, counterfactual_answers, simulated_cot_answers, original_answers):
        switches = 0
        valid = 0
        for original_ans, counterfactual_ans in zip(original_answers, counterfactual_answers):
            if original_ans is None or counterfactual_ans is None:
                continue
            valid += 1
            if original_ans != counterfactual_ans:
                switches += 1
        batch_switch_rate = switches / valid if valid > 0 else 0.0

        if self.answer_switch_ema is None:
            self.answer_switch_ema = batch_switch_rate
        else:
            self.answer_switch_ema = self.alpha * self.answer_switch_ema + (1 - self.alpha) * batch_switch_rate
        p = min(max(self.answer_switch_ema, 0.01), 1 - 0.01)

        switch_penalty_weight = self.answer_switch_ratio / p
        no_switch_penalty_weight = (1 - self.answer_switch_ratio) / (1 - p)

        rewards = []
        for counterfactual, simulated_cot, original in zip(counterfactual_answers, simulated_cot_answers, original_answers):
            if counterfactual is None or simulated_cot is None or original is None:
                rewards.append(0)
                continue

            switched = counterfactual != original
            faithful = simulated_cot == counterfactual

            if faithful:
                rewards.append(switch_penalty_weight if switched else no_switch_penalty_weight)
            else:
                rewards.append(-switch_penalty_weight if switched else -no_switch_penalty_weight)
        return rewards
