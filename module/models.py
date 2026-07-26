import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class Model:
    def __init__(
        self,
        name,
        max_tokens=256,
        temperature=0.7,
        devices=None
    ):
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.devices = devices.split(",") if devices is not None else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = LLM(
            model=name,
            tensor_parallel_size=torch.cuda.device_count() if self.devices is None else len(self.devices),
            dtype=torch.bfloat16,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
        )

    def generate_response(self, prompt, thinking=False, n_completions=1, **kwargs):
        responses = self.batch_generate_response([prompt], thinking=thinking, n_completions=n_completions, **kwargs)
        return responses

    def batch_generate_response(self, prompts, thinking=False, n_completions=1):
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking
            )
            for prompt in prompts
        ]

        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": n_completions,
        }

        sampling_params = SamplingParams(**params)

        all_outputs = []

        results = self.llm.generate(texts, sampling_params)
        for r in results:
            completions = [
                o.text.strip() for o in r.outputs
            ]
            all_outputs.extend(completions)
        return all_outputs
