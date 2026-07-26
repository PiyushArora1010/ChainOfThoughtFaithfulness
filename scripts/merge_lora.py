import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

adapter_path = sys.argv[1]
save_path = adapter_path

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
)
model = PeftModel.from_pretrained(
    model,
    adapter_path,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

model = model.merge_and_unload()

output_dir = os.path.join(save_path, "model")
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)