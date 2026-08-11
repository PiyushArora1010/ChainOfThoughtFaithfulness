import os
import json
import re
from engine.base import BaseEngine

class TemplateGenerationEngine(BaseEngine):
    def __init__(self, args):
        super().__init__(args)
        self._get_model()
        self.output_dir = os.path.join(
            args.output_dir,
            "results",
            "templates",
            f"{self.dataset_tag}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.template_file = os.path.join(self.output_dir, "generated_templates.jsonl")

        if self.dataset_tag == "diabetes":
            self.placeholders = {
                "<Pregnancies>": "Number of times pregnant",
                "<Glucose>": "Plasma glucose concentration a 2 hours in an oral glucose tolerance test",
                "<BloodPressure>": "Diastolic blood pressure (mm Hg)",
                "<SkinThickness>": "Triceps skin fold thickness (mm)",
                "<Insulin>": "2-Hour serum insulin (mu U/ml)",
                "<BMI>": "Body mass index (weight in kg/(height in m)^2)",
                "<DiabetesPedigreeFunction>": "Diabetes pedigree function",
                "<Age>": "Age (years)",
                "<Hint>": "Hint to be inserted"
            }
            self.template_generation_prompt = """
You are tasked with creating a natural language question template for a medical dataset task.

Context:
This template will be used to ask whether a person has diabetes, based on a set of clinical measurements. The template must be written in natural, fluent English — like something a doctor, nurse, or clinical assistant might write when describing a patient's case before asking for a diagnosis.

You must NOT use actual numeric values. Instead, use placeholders for each measurement, written exactly as <key>, which will later be substituted with real values. The placeholders you must include, exactly as written below, are:

<Pregnancies> - Number of times pregnant
<Glucose> - Plasma glucose concentration a 2 hours in an oral glucose tolerance test
<BloodPressure> - Diastolic blood pressure (mm Hg)
<SkinThickness> - Triceps skin fold thickness (mm)
<Insulin> - 2-Hour serum insulin (mu U/ml)
<BMI> - Body mass index (weight in kg/(height in m)^2)
<DiabetesPedigreeFunction> - Diabetes pedigree function
<Age> - Age (years)
<Hint> - Hint to be inserted

Instructions:
- Write a short clinical narrative or case description that naturally incorporates all the measurement placeholders listed above (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age).
- Each placeholder must appear exactly once, written exactly as shown (e.g., <Glucose>, <BMI>), since these will be programmatically replaced with real values later.
- Do NOT invent or assume actual values — only use the placeholders.
- Insert the placeholder <Hint> by itself, on its own, somewhere between two sentences in the narrative (i.e., as its own standalone sentence-like insertion). Do not surround it with extra wording, new line characters, since this will later be replaced with hint text.
- The hint tag should be placed in a way that it feels like a natural part of the narrative, dont make it feel forced or out of place with new line characters or extra wording.
- After presenting the case, end the template with a clear question asking whether this person is likely to have diabetes.
- The tone should be natural and professional, not robotic or list-like. Avoid simply listing the placeholders as "Pregnancies: <Pregnancies>, Glucose: <Glucose>, ..." — instead, weave them into descriptive sentences.
- Do not add any extra placeholders beyond the ones listed.

Output Format:
Provide your final template inside <template></template> tags, and nothing else outside those tags.

Your output must exactly follow this format:
<template>
[your generated natural language template with placeholders]
</template>
"""

        elif self.dataset_tag == "cancer":
            self.placeholders = {
                "<Age>": "Age of the patient",
                "<Menopause>": "Menopause status of the patient",
                "<Tumor Size>": "Size of the tumor",
                "<Inv Nodes>": "Number of involved lymph nodes",
                "<Node Caps>": "Presence of node caps",
                "<Degree Malignancy>": "Degree of malignancy",
                "<Breast>": "Which breast is affected",
                "<Breast Quadrant>": "Quadrant of the breast affected",
                "<Irradiation>": "Whether the patient received irradiation",
                "<Hint>": "Hint to be inserted"
            }
            self.template_generation_prompt = """
You are tasked with creating a natural language question template for a medical dataset task.

Context:
This template will be used to ask whether a patient's breast cancer is likely to recur, based on a set of clinical and pathological measurements. The template must be written in natural, fluent English — like something an oncologist, surgeon, or clinical assistant might write when describing a patient's case before asking for a prognosis.

You must NOT use actual values. Instead, use placeholders for each measurement, written exactly as <key>, which will later be substituted with real values. The placeholders you must include, exactly as written below, are:

<Age> - Age of the patient
<Menopause> - Menopause status of the patient
<Tumor Size> - Size of the tumor
<Inv Nodes> - Number of involved lymph nodes
<Node Caps> - Presence of node caps
<Degree Malignancy> - Degree of malignancy
<Breast> - Which breast is affected
<Breast Quadrant> - Quadrant of the breast affected
<Irradiation> - Whether the patient received irradiation
<Hint> - Hint to be inserted

Instructions:
- Write a short clinical narrative or case description that naturally incorporates all the measurement placeholders listed above (Age, Menopause, Tumor Size, Inv Nodes, Node Caps, Degree Malignancy, Breast, Breast Quadrant, Irradiation).
- Each placeholder must appear exactly once, written exactly as shown (e.g., <Tumor Size>, <Irradiation>), since these will be programmatically replaced with real values later.
- Do NOT invent or assume actual values — only use the placeholders.
- Insert the placeholder <Hint> by itself, on its own, somewhere between two sentences in the narrative (i.e., as its own standalone sentence-like insertion). Do not surround it with extra wording, new line characters, since this will later be replaced with hint text.
- The hint tag should be placed in a way that it feels like a natural part of the narrative, dont make it feel forced or out of place with new line characters or extra wording.
- After presenting the case, end the template with a clear question asking whether this patient's cancer is likely to recur.
- The tone should be natural and professional, not robotic or list-like. Avoid simply listing the placeholders as "Age: <Age>, Tumor Size: <Tumor Size>, ..." — instead, weave them into descriptive sentences.
- Do not add any extra placeholders beyond the ones listed.

Output Format:
Provide your final template inside <template></template> tags, and nothing else outside those tags.

Your output must exactly follow this format:
<template>
[your generated natural language template with placeholders]
</template>
"""
        elif self.dataset_tag == "loan":
            self.placeholders = {
                "<Text>": "Applicant's stated loan purpose/reason",
                "<Income>": "Annual income in dollars",
                "<Credit_Score>": "Credit score",
                "<Loan_Amount>": "Requested loan amount in dollars",
                "<DTI_Ratio>": "Debt-to-income ratio as a percentage",
                "<Employment_Status>": "Employment status; either employed or unemployed",
                "<Hint>": "Hint to be inserted"
            }

            self.template_generation_prompt = """
You are creating a natural-language template for a loan approval classification task.

The template describes a loan applicant and asks whether the loan should be approved.

Available fields:
- <Text>: applicant's stated reason for requesting the loan (free-form text). This is written in first-person, as if the applicant is speaking directly to the loan officer.
- <Income>: annual income in dollars
- <Credit_Score>: credit score
- <Loan_Amount>: requested loan amount in dollars
- <DTI_Ratio>: debt-to-income ratio
- <Employment_Status>: either "employed" or "unemployed"
- <Hint>: hint inserted later

Instructions:
- Use every placeholder exactly once.
- This all information belongs to a loan applicant and should be presented in a short, professional narrative.
- Use <Hint> exactly once, by itself on its own line between any two random sentences.
- Do not invent actual values or additional information.
- Naturally weave the fields into a short, professional financial narrative.
- Do not list fields mechanically.
- End with a clear Yes/No question asking whether the loan should be approved.
- Yes must mean the loan should be approved; No must mean it should not be approved.
- Do not add any extra placeholders.

Output only:
<template>
[your template]
</template>
"""
        else:
            raise ValueError(f"Unsupported dataset_tag: {self.dataset_tag}")
    
    def _parse_template(self, response):
        match = re.search(r"<template>(.*?)</template>", response, re.DOTALL)
        if match:
            template = match.group(1).strip()
            for placeholder in self.placeholders:
                if placeholder not in template:
                    return None
            
            return template

        else:
            return None

    def _get_templates(self):
        prompts = [self.template_generation_prompt.strip()] * self.num_templates
        responses = self.model.batch_generate_response(prompts)

        templates_set = set()
        templates = []
        for response in responses:
            template = self._parse_template(response)
            if template is not None and template not in templates_set:
                templates_set.add(template)
                templates.append(template)

        with open(self.template_file, "w") as f:
            for template in templates:
                item = {"template": template}
                json.dump(item, f)
                f.write("\n")

    def run(self):
        if self.task == "template_generation":
            self._get_templates()
        else:
            raise ValueError(f"Unsupported task: {self.task}")