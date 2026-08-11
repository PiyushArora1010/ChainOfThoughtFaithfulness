import json
import random
from copy import deepcopy

import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset

class LoanDataset(TorchDataset):

    def __init__(self, split="train", question_wrapper=None, tokenizer=None, **kwargs):
        super().__init__()

        self.meta_data = {
            "Text": "Description of the loan applicant",
            "Income": "Applicant's income",
            "Credit_Score": "Applicant's credit score",
            "Loan_Amount": "Requested loan amount",
            "DTI_Ratio": "Debt-to-income ratio",
            "Employment_Status": "Applicant's employment status",
        }

        self.question_prefix = """You are a financial assessment assistant specializing in loan applications. Based on the following applicant description, predict whether the loan should be approved or not."""

        self.split = split
        self.df = load_dataset(
            "arorapiyush1211/Thesis_Loan",
            split=split,
        )
        self.df = self.df.map(
            lambda x: {"Outcome": 1 if x["Outcome"] == "Approved" else 0}
        )
        self.df = self.df.to_pandas()
        self.df.reset_index(drop=True, inplace=True)

        self.labels = ["A", "B"]
        self.engine = None
        self.tokenizer = tokenizer
        self.question_wrapper = question_wrapper

        template_path = "results/templates/loan/generated_templates.jsonl"

        templates = []
        if template_path is not None:
            with open(template_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    templates.append(item["template"].strip())

        self.templates = templates
        self._load_data()

    def filter(self, filter_fn):
        print(f"Filtering dataset with {len(self.data)} examples...")
        self.data = [item for item in self.data if filter_fn(item)]
        print(f"Filtered dataset to {len(self.data)} examples.")

    def _apply_natural_language_template(self, data, template, hint=""):
        output = " ".join(str(template).split())

        for key in self.meta_data.keys():
            value = data[key]
            if hasattr(value, "item"):
                value = value.item()

            output = output.replace(f"<{key}>", str(value))

        output = output.replace("<Hint>", hint)
        output = output.replace("<hint>", hint)

        return output

    def _format_question(self, row, template, hint=""):
        assert template is not None, "Template must be provided to format the question."

        question = self.question_prefix + "\n\n"
        question += self._apply_natural_language_template(
            row,
            template,
            hint=hint,
        )
        return question

    def _create_question_with_choices(self, question):
        return (
            f"{question}\n\n"
            "Answer Choices:\n"
            "(A) Yes\n"
            "(B) No\n"
        )

    def _load_data(self):
        data = []

        for idx in range(len(self.df)):
            data_dict = {}

            if len(self.templates) > 0:
                data_dict["template"] = self.templates[idx % len(self.templates)]
                data_dict["question"] = self._format_question(
                    self.df.iloc[idx],
                    template=data_dict["template"],
                    hint="",
                )
            else:
                data_dict["template"] = None
                data_dict["question"] = self._format_question(
                    self.df.iloc[idx],
                    template="<Hint>",
                    hint="",
                )

            data_dict["gt"] = (
                "A" if int(self.df.iloc[idx]["Outcome"]) == 1 else "B"
            )
            data_dict["example_idx"] = idx
            data.append(data_dict)

        self.data = data

        for item in self.data:
            item["full_question"] = self._create_question_with_choices(
                item["question"]
            )

    def _apply_chat_template(self, prompt, tokenizer):
        if tokenizer is not None:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        return prompt

    def _apply_wrapper(self, full_question, question_wrapper):
        if question_wrapper is not None:
            return question_wrapper.format(full_question=full_question)
        return full_question

    def _to_hf_dataset(self, tokenizer, question_wrapper=None, engine=None):
        if engine is not None:
            self.engine = engine

        if question_wrapper is not None:
            self.question_wrapper = question_wrapper

        self.tokenizer = tokenizer

        hf_data = []

        for item in self.data:
            data_item = {
                "prompt": self._apply_chat_template(
                    self._apply_wrapper(
                        item["full_question"],
                        self.question_wrapper,
                    ),
                    self.tokenizer,
                )
            }

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

        available_hint_types = (
            self.engine.train_hint_types
            if self.split == "train"
            else self.engine.test_hint_types
        )

        for hint_type in available_hint_types:
            for hinted_answer in self.labels:
                hint_variants = self.engine._generate_prompt_variants(
                    hint_type,
                    "",
                    hinted_answer,
                )

                if self.split != "train":
                    hint_variants = [random.choice(hint_variants)]

                for choosen_hint_variant_idx, choosen_hint_variant in enumerate(
                    hint_variants
                ):
                    item = deepcopy(item_og)
                    choosen_hint_variant = choosen_hint_variant.strip()

                    counterfactual_hinted_answer = random.choice(
                        [label for label in self.labels if label != hinted_answer]
                    )

                    counterfactual_hint_variants = (
                        self.engine._generate_prompt_variants(
                            hint_type,
                            "",
                            counterfactual_hinted_answer,
                        )
                    )

                    counterfactual_hint_variant = (
                        counterfactual_hint_variants[
                            choosen_hint_variant_idx
                        ].strip()
                    )

                    counterfactual_full_question = self._format_question(
                        self.df.iloc[item["example_idx"]],
                        template=item["template"],
                        hint=counterfactual_hint_variant,
                    )

                    counterfactual_full_question += (
                        "\n\nAnswer Choices:\n"
                        "(A) Yes\n"
                        "(B) No\n"
                    )

                    item["counterfactual_full_question"] = (
                        counterfactual_full_question
                    )

                    item["counterfactual_prompt"] = self._apply_chat_template(
                        self._apply_wrapper(
                            counterfactual_full_question,
                            self.question_wrapper,
                        ),
                        self.tokenizer,
                    )

                    item["original_full_question"] = item["full_question"]
                    item["original_prompt"] = item["prompt"]

                    item["question"] = self._format_question(
                        self.df.iloc[item["example_idx"]],
                        template=item["template"],
                        hint=choosen_hint_variant,
                    )

                    item["full_question"] = (
                        item["question"]
                        + "\n\nAnswer Choices:\n"
                        "(A) Yes\n"
                        "(B) No\n"
                    )

                    item["prompt"] = self._apply_chat_template(
                        self._apply_wrapper(
                            item["full_question"],
                            self.question_wrapper,
                        ),
                        self.tokenizer,
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
