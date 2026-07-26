import os
from unsloth import FastLanguageModel
import torch
import asyncio
from openai import AsyncOpenAI

from engine.base import BaseEngine

class TrainExplanationEngine(BaseEngine):
    def __init__(self, args):
        super().__init__(args)
        self.output_dir = os.path.join(
            args.output_dir,
            "results",
            "training_explanation",
            self.dataset_tag,
            args.run_name
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.implied_client = None
        self._get_implied_client()

        self.if_influenced_by_hint_prompt = """
"""
        self.get_counterfactual_prompt = """
"""

    def _get_implied_client(self):
        self.implied_client = AsyncOpenAI(
            api_key="EMPTY",  # vLLM ignores this
            base_url=self.implied_model_url,
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

    # MAIN MODEL
    def _get_model_and_tokenizer(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_tag,
            max_seq_length=self.model_max_tokens,
            load_in_4bit=False,  # False for LoRA 16bit
            fast_inference=self.use_vllm,
            max_lora_rank=self.lora_rank,
            gpu_memory_utilization = 0.5,
            dtype = torch.float16
        )
        
        if self.lora:
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules=self.lora_layers,  # Remove QKVO if out of memory
                lora_alpha=self.lora_rank*2,
                use_gradient_checkpointing=False,  # Enable long context finetuning
                random_state=self.seed
            )
        model.generation_config.max_new_tokens = self.model_max_tokens
        model.generation_config.temperature = self.model_temperature
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer
    
    def _prepare_datasets(self, tokenizer):
        self.dataset._to_hf_dataset(
            tokenizer=tokenizer,
            question_wrapper=self.base_prompt_answer,
            engine=self
        )

        train_size = len(self.dataset) - 100
        val_size = 100
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
            max_length=self.model_max_tokens,
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=self.model_max_tokens,
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

    def _get_answers_from_responses(self, responses):
        return [self._parse_answer(response) for response in responses]

    def _get_reasoning_from_responses(self, responses):
        return [self._parse_reasoning(response) for response in responses]

    