import argparse
from engine.hint_faithfulness import HintFaithfulnessEngine

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--dataset_tag', type=str, default='diabetes')
parser.add_argument('--dataset_path', type=str, default='module/datasets/data/diabetes_with_split.csv')
parser.add_argument('--template_path', type=str, default=None)
parser.add_argument('--concept_cf', action='store_true')
parser.add_argument('--hint_cf', action='store_true')
parser.add_argument('--split', type=str, default='test')

# task
parser.add_argument('--task', type=str, default='responses,implied_responses,faithfulness')
parser.add_argument('--counterfactual_question_key', type=str, default='counterfactual')

# model settings
parser.add_argument('--model_tag', type=str, default='Qwen/Qwen3-32B')
parser.add_argument('--model_max_tokens', type=int, default=512)
parser.add_argument('--model_thinking', action='store_true')
parser.add_argument('--model_top_p', type=float, default=None)
parser.add_argument('--model_top_k', type=int, default=None)
parser.add_argument('--model_min_p', type=float, default=None)
parser.add_argument('--model_temperature', type=float, default=0.0)

# directories
parser.add_argument('--save_tag', type=str, default='default')
parser.add_argument('--output_dir', type=str, default='output')

# random seed
parser.add_argument('--seed', type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    engine = HintFaithfulnessEngine(args)
    engine.run()