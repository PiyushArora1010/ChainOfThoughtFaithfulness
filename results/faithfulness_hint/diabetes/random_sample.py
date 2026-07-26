# import json
# import random

# # -----------------------------
# # Load JSON files
# # -----------------------------
# with open("/home/piyush/Desktop/Code/SandbarASR/CueFaithfulness/results/faithfulness_hint/diabetes/Qwen3_4B_ema_500/implied_hint_results.jsonl", "r") as f:
#     data1 = json.load(f)

# with open("/home/piyush/Desktop/Code/SandbarASR/CueFaithfulness/results/faithfulness_hint/diabetes/Qwen3_4B_hint_equals_gt_stage2_1000/implied_hint_results.jsonl", "r") as f:
#     data2 = json.load(f)

import json
import random

FILE1 = "/home/piyush/Desktop/Code/SandbarASR/CueFaithfulness/results/faithfulness_hint/diabetes/Qwen3_4B_ema_500/implied_hint_results.jsonl"
FILE2 = "/home/piyush/Desktop/Code/SandbarASR/CueFaithfulness/results/faithfulness_hint/diabetes/Qwen3_4B_hint_equals_gt_stage2_1000/implied_hint_results.jsonl"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Load files
data1 = load_jsonl(FILE1)
data2 = load_jsonl(FILE2)

# Build lookup dictionaries by example_idx
dict1 = {item["example_idx"]: item for item in data1}
dict2 = {item["example_idx"]: item for item in data2}

# Matching example_idx from file1
file1_idx = {
    item["example_idx"]
    for item in data1
    if item["original_pred"] != item["pred"]
    and str(item["verdict"]).strip().lower() == "no"
}

# Matching example_idx from file2
file2_idx = {
    item["example_idx"]
    for item in data2
    if item["original_pred"] == item["pred"]
    and str(item["verdict"]).strip().lower() == "yes"
}

# Common example_idx
common_idx = list(file1_idx & file2_idx)

print(f"Found {len(common_idx)} common examples.")

if not common_idx:
    print("No common samples found.")
    exit()

# Pick one random example
idx = random.choice(common_idx)

sample1 = dict1[idx]
sample2 = dict2[idx]

print("=" * 100)
print(f"Example ID: {idx}")
print("=" * 100)

print("\nQuestion:")
print(sample1.get("question", sample1.get("input", "N/A")))

print("\nGround Truth:", sample1.get("label"))
print("Original Pred:", sample1.get("original_pred"))
print("File1 Pred:", sample1.get("pred"))
print("File2 Pred:", sample2.get("pred"))

print("\n" + "=" * 100)
print("COT FROM FILE 1")
print("=" * 100)
print(sample1.get("cot", "No cot field found."))

print("\n" + "=" * 100)
print("COT FROM FILE 2")
print("=" * 100)
print(sample2.get("cot", "No cot field found."))