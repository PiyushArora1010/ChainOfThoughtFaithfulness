# arorapiyush1211/Thesis_SNLI
import json
import random
from copy import deepcopy

import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset


class SNLIDataset(TorchDataset):
    """
    Loads arorapiyush1211/Thesis_SNLI, which has columns:
        premise, hypothesis, Outcome   (Outcome: 1 = entailment, 0 = contradiction)
    and splits: "train", "test" (no per-task configs, unlike the Ethics loader
    this class was adapted from).
    """

    def __init__(self, split="train", question_wrapper=None, tokenizer=None, **kwargs):
        super().__init__()

        self.hf_name = "arorapiyush1211/Thesis_SNLI"
        template_path = None

        self.meta_data = {
            "Premise": "The premise sentence describing a situation.",
            "Hypothesis": "A statement to be judged against the premise.",
        }

        self.bins = {}

        self.question_prefix = """You are a natural language inference assistant. Given a premise and a hypothesis, determine whether the hypothesis is entailed by the premise or whether it contradicts the premise."""

        self.question_templates = {
            "snli": "Based on the above premise and hypothesis, is the hypothesis entailed by the premise?",
        }

        self.split = split
        self._hf_split = "test" if split in ("test", "validation") else "train"

        self.df = self._load_hf_snli()

        self.labels = ["A", "B"]
        self.engine = None
        self.tokenizer = tokenizer
        self.question_wrapper = question_wrapper

        templates = []
        if template_path is not None:
            with open(template_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    templates.append(item["template"].strip())
        self.templates = templates

        self._load_data()

    def _load_hf_snli(self):
        ds = load_dataset(self.hf_name, split=self._hf_split)
        df = ds.to_pandas()
        return df[["premise", "hypothesis", "Outcome"]].reset_index(drop=True)

    def filter(self, filter_fn):
        print(f"Filtering dataset with {len(self.data)} examples...")
        self.data = [item for item in self.data if filter_fn(item)]
        print(f"Filtered dataset to {len(self.data)} examples.")

    def _apply_natural_language_template(self, data, template, hint=""):
        output = str(template)
        output = " ".join(output.split())

        for key in self.meta_data.keys():
            value = data[key.lower()]
            value = value.item() if hasattr(value, "item") else value
            output = output.replace(f"<{key}>", str(value))
        output = output.replace("<Hint>", hint)
        output = output.replace("<hint>", hint)
        if not output.endswith(" "):
            output += " "
        output += self.question_templates["snli"]
        return output

    def _format_question(self, row, template, hint=""):
        assert template is not None, "Template must be provided to format the question."
        question = self.question_prefix + "\n\n"
        question += self._apply_natural_language_template(row, template, hint=hint)
        return question

    def _create_question_with_choices(self, question):
        return f"{question}\n\nAnswer Choices:\n(A) Yes\n(B) No\n"

    def _load_data(self):
        data = []
        default_template = [
            "\nPremise: \"<Premise>\"\nHypothesis: \"<Hypothesis>\"\n<Hint>",
            "\n<Hint>\nPremise: \"<Premise>\"\nHypothesis: \"<Hypothesis>\""
        ]

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            data_dict = {}

            if len(self.templates) > 0:
                data_dict["template"] = self.templates[idx % len(self.templates)]
            else:
                data_dict["template"] = random.choice(default_template)

            data_dict["question"] = self._format_question(row, template=data_dict["template"], hint="")
            data_dict["gt"] = "A" if row["Outcome"] == 1 else "B"
            data_dict["example_idx"] = idx

            data.append(data_dict)

        self.data = data

        for idx, item in enumerate(self.data):
            item["full_question"] = self._create_question_with_choices(item["question"])

    def _apply_chat_template(self, prompt, tokenizer):
        if tokenizer is not None:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            return prompt

    def _apply_wrapper(self, full_question, question_wrapper):
        if question_wrapper is not None:
            return question_wrapper.format(full_question=full_question)
        else:
            return full_question

    def _to_hf_dataset(self, tokenizer, question_wrapper=None, engine=None):

        if engine is not None:
            self.engine = engine

        if question_wrapper is not None:
            self.question_wrapper = question_wrapper

        self.tokenizer = tokenizer

        hf_data = []

        for item in self.data:
            data_item = {}

            data_item["prompt"] = self._apply_chat_template(
                self._apply_wrapper(
                    item["full_question"],
                    self.question_wrapper
                ),
                self.tokenizer
            )

            for key, value in item.items():
                data_item[key] = value

            hf_data.append(data_item)

        self.data = hf_data

        if engine is not None:
            if engine.hint_cf:
                print(f"Generating hinted items for {len(self.data)} examples...")
                hinted_items = []
                for idx in range(len(self.data)):
                    hinted_items.extend(self._get_hinted_items(idx))
                print(f"Generated {len(hinted_items)} hinted items.")
            else:
                hinted_items = []

            self.data = hinted_items

            random.shuffle(self.data)

    def _get_hinted_items(self, idx):
        item_og = self.data[idx]

        hinted_items = []

        available_hint_types = self.engine.train_hint_types if self.split == "train" else self.engine.test_hint_types

        for hint_type in available_hint_types:
            for hinted_answer in self.labels:

                hint_variants = self.engine._generate_prompt_variants(
                    hint_type,
                    "",
                    hinted_answer
                )

                if self.split != "train":
                    hint_variants = [random.choice(hint_variants)]

                for choosen_hint_variant_idx, choosen_hint_variant in enumerate(hint_variants):
                    item = deepcopy(item_og)

                    choosen_hint_variant = choosen_hint_variant.strip()

                    counterfactual_hinted_answer = random.choice([label for label in self.labels if label != hinted_answer])
                    counterfactual_hint_variant = self.engine._generate_prompt_variants(
                        hint_type,
                        "",
                        counterfactual_hinted_answer
                    )[choosen_hint_variant_idx].strip()

                    counterfactual_full_question = self._format_question(
                        self.df.iloc[item["example_idx"]],
                        template=item["template"],
                        hint=counterfactual_hint_variant
                    )
                    counterfactual_full_question += "\n\nAnswer Choices:\n(A) Yes\n(B) No\n"

                    item["counterfactual_full_question"] = counterfactual_full_question
                    item["counterfactual_prompt"] = self._apply_chat_template(
                        self._apply_wrapper(
                            counterfactual_full_question,
                            self.question_wrapper
                        ),
                        self.tokenizer
                    )

                    item["original_full_question"] = item["full_question"]
                    item["original_prompt"] = item["prompt"]

                    item["question"] = self._format_question(
                        self.df.iloc[item["example_idx"]],
                        template=item["template"],
                        hint=choosen_hint_variant
                    )
                    item["full_question"] = item["question"] + "\n\nAnswer Choices:\n(A) Yes\n(B) No\n"
                    item["prompt"] = self._apply_chat_template(
                        self._apply_wrapper(
                            item["full_question"],
                            self.question_wrapper
                        ),
                        self.tokenizer
                    )
                    item["hint_type"] = hint_type
                    item["hinted_answer"] = hinted_answer
                    item["inserted_hint"] = choosen_hint_variant

                    hinted_items.append(item)

        return hinted_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    dataset = SNLIDataset(split="test", tokenizer=None, question_wrapper=None)
    breakpoint()