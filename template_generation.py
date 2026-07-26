import argparse
from engine.template_generation import TemplateGenerationEngine

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--dataset_tag', type=str, default='diabetes')
parser.add_argument('--dataset_path', type=str, default='module/datasets/data/diabetes_with_split.csv')
parser.add_argument('--template_path', type=str, default=None)
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--num_templates', type=int, default=2000, help="Number of templates")

# task
parser.add_argument('--task', type=str, default='template_generation')

# model settings
parser.add_argument('--model_tag', type=str, default='Qwen/Qwen3-32B')
parser.add_argument('--model_max_tokens', type=int, default=512)
parser.add_argument('--model_thinking', action='store_true')
parser.add_argument('--model_top_p', type=float, default=None)
parser.add_argument('--model_top_k', type=int, default=None)
parser.add_argument('--model_min_p', type=float, default=None)
parser.add_argument('--model_temperature', type=float, default=1.0)

# directories
parser.add_argument('--output_dir', type=str, default='output')

# random seed
parser.add_argument('--seed', type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    engine = TemplateGenerationEngine(args)
    engine.run()