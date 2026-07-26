import json
import pandas as pd
import numpy as np
import random

from module.datasets.counterfactuals import add_counterfactuals

from torch.utils.data import Dataset as TorchDataset
from copy import deepcopy

class DiabetesDataset(TorchDataset):
    def __init__(self, dataset_path, template_path=None, split="train", question_wrapper=None, tokenizer=None, **kwargs):
        super().__init__()
        assert template_path is not None, "Template path must be provided."
        self.meta_data = {
            "Pregnancies": "Number of times pregnant",
            "Glucose": "Plasma glucose concentration a 2 hours in an oral glucose tolerance test",
            "BloodPressure": "Diastolic blood pressure (mm Hg)",
            "SkinThickness": "Triceps skin fold thickness (mm)",
            "Insulin": "2-Hour serum insulin (mu U/ml)",
            "BMI": "Body mass index (weight in kg/(height in m)^2)",
            "DiabetesPedigreeFunction": "Diabetes pedigree function",
            "Age": "Age (years)"
        }

        self.question_prefix = """You are a medical assessment assistant specializing in diabetes risk. Based on the following patient description, predict whether the patient has diabetes or not."""
        self.bins = {
            "Pregnancies": ([-np.inf, 0, 2, 5, np.inf], ["0", "1-2", "3-5", "6+"]),
            "Glucose": ([-np.inf, 100, 126, np.inf], ["Normal", "Prediabetic", "Diabetic"]),
            "BloodPressure": ([-np.inf, 80, 90, np.inf], ["Normal", "Elevated", "High"]),
            "SkinThickness": ([-np.inf, 10, 20, 30, np.inf], ["Low", "Below Average", "Average", "High"]),
            "Insulin": ([-np.inf, 16, 166, np.inf], ["Low", "Normal", "High"]),
            "BMI": ([-np.inf, 18.5, 25, 30, np.inf], ["Underweight", "Normal", "Overweight", "Obese"]),
            "DiabetesPedigreeFunction": ([-np.inf, 0.2, 0.5, 1.0, np.inf], ["Low", "Below Average", "Average", "High"]),
            "Age": ([-np.inf, 24, 54, 64, np.inf], ["15-24", "25-54", "55-64", "65+"]),
        }

        self.split = split
        self.df = pd.read_csv(dataset_path)
        self.df = self.df[self.df["split"] == split]
        self.df.reset_index(drop=True, inplace=True)

        self.df = add_counterfactuals(self.df, self.bins, label_col="Outcome", r=2, m=10, eps=0.3, seed=0)

        self.labels = ["A", "B"]
        self.engine = None  # Placeholder for the engine instance
        self.tokenizer = tokenizer  # Placeholder for the tokenizer instance
        self.question_wrapper = question_wrapper  # Placeholder for the question wrapper

        templates = []
        if template_path is not None:
            with open(template_path, 'r') as f:
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
        output = str(template)
        output = " ".join(output.split())

        for key in self.meta_data.keys():
            output = output.replace(
                f"<{key}>", str(data[key].item())
            )
        output = output.replace("<Hint>", hint)
        output = output.replace("<hint>", hint)
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

        for idx in range(len(self.df)):
            data_dict = {}
            if len(self.templates) > 0:
                data_dict["template"] = self.templates[idx % len(self.templates)] if self.templates else None
                data_dict["question"] = self._format_question(self.df.iloc[idx], template=data_dict["template"], hint="")
            else:
                print("No templates provided. This is recommened only for template generation.")
                data_dict["template"] = None
                data_dict["question"] = self._format_question(self.df.iloc[idx], template="<Hint>", hint="")

            data_dict["gt"] = "A" if self.df.iloc[idx]["Outcome"] == 1 else "B"
            data_dict["example_idx"] = idx

            data.append(data_dict)

        self.data = data

        for idx, item in enumerate(self.data):
            item["full_question"] = self._create_question_with_choices(item["question"])

    # Convert to HF dataset format with prompts and counterfactuals

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
            self.engine = engine  # Store the engine instance for later use

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

            if engine.concept_cf:
                print(f"Generating counterfactual items for {len(self.data)} examples...")
                counterfactual_items = []
                for idx in range(len(self.data)):
                    counterfactual_items.extend(self._get_counterfactual_items(idx))
                print(f"Generated {len(counterfactual_items)} counterfactual items.")
            else:
                counterfactual_items = []

            if engine.hint_cf and engine.concept_cf:
                ratio_hint = engine.hint_cf_ratio

                random.shuffle(hinted_items)
                random.shuffle(counterfactual_items)

                n_hint = len(hinted_items)
                n_cf = len(counterfactual_items)

                max_total = min(
                    n_hint / ratio_hint if ratio_hint > 0 else float("inf"),
                    n_cf / (1 - ratio_hint) if ratio_hint < 1 else float("inf"),
                )

                final_hint = min(int(round(max_total * ratio_hint)), n_hint)
                final_cf = min(int(round(max_total * (1 - ratio_hint))), n_cf)

                print(f"Using {final_hint} hinted items and {final_cf} counterfactual items for training.")

                self.data = hinted_items[:final_hint] + counterfactual_items[:final_cf]
            else:
                self.data = hinted_items + counterfactual_items

            random.shuffle(self.data)

    def _get_counterfactual_items(self, idx):
        item_og = self.data[idx]
        example_idx = item_og["example_idx"]

        counterfactual_items = []
        for cf_idx, diff_cols in self.df.loc[example_idx, "counterfactuals"]:
            item = deepcopy(item_og)  # Create a copy of the original item

            cf_row = self.df.iloc[cf_idx]
            original_row = self.df.iloc[example_idx]

            data_for_cf = {}
            for key in self.meta_data.keys():
                if key in diff_cols:
                    data_for_cf[key] = cf_row[key]
                else:
                    data_for_cf[key] = original_row[key]
            
            item["counterfactual_full_question"] = f"{self._format_question(data_for_cf, item["template"])}\n\nAnswer Choices:\n(A) Yes\n(B) No\n"
            item["counterfactual_prompt"] = self._apply_chat_template(
                self._apply_wrapper(
                    item["counterfactual_full_question"],
                    self.question_wrapper
                ),
                self.tokenizer
            )

            item["original_full_question"] = item["full_question"] # Store the original full_question before modification
            item["original_prompt"] = item["prompt"] # Use the original prompt as the counterfactual prompt

            item["hint_type"] = "concept"
            item["hinted_answer"] = None
            item["inserted_hint"] = diff_cols  # Store the differing columns as the hint


            counterfactual_items.append(item)

        return counterfactual_items

    def _get_hinted_items(self, idx):
        item_og = self.data[idx]

        hinted_items = []

        available_hint_types = self.engine.train_hint_types if self.split == "train" else self.engine.test_hint_types

        for hint_type in available_hint_types:
            for hinted_answer in self.labels:    
                # if hinted_answer != item_og["gt"] and self.split == "train":
                #     if random.random() < 0.75:
                #         continue  # Skip if the hinted answer is not the ground truth during training

                hint_variants = self.engine._generate_prompt_variants(
                    hint_type,
                    "",
                    hinted_answer
                )

                if self.split != "train":
                    hint_variants = [random.choice(hint_variants)]

                for choosen_hint_variant_idx, choosen_hint_variant in enumerate(hint_variants):
                    item = deepcopy(item_og)  # Create a copy of the original item

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

                    item["counterfactual_full_question"] = counterfactual_full_question #item["full_question"] # Store the original full_question before modification
                    item["counterfactual_prompt"] = self._apply_chat_template(
                        self._apply_wrapper(
                            counterfactual_full_question,
                            self.question_wrapper
                        ),
                        self.tokenizer
                    )

                    item["original_full_question"] = item["full_question"] # Store the original full_question before modification
                    item["original_prompt"] = item["prompt"] # Use the original prompt as the counterfactual prompt?

                    item["question"] = self._format_question(
                        self.df.iloc[item["example_idx"]],
                        template=item["template"],
                        hint=choosen_hint_variant
                    ) 
                    item["full_question"] = item["question"] + "\n\nAnswer Choices:\n(A) Yes\n(B) No\n" # Update the full_question with the hinted variant
                    item["prompt"] = self._apply_chat_template(
                        self._apply_wrapper(
                            item["full_question"],
                            self.question_wrapper
                        ),
                        self.tokenizer
                    )
                    item["hint_type"] = hint_type
                    item["hinted_answer"] = hinted_answer
                    item["inserted_hint"] = choosen_hint_variant  # Store the inserted hint

                    hinted_items.append(item)

        return hinted_items

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    dataset = DiabetesDataset("data/diabetes_with_split.csv", split="train")
    breakpoint()