import json
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with open("data/train.jsonl", encoding="utf-8") as file:
    example = json.loads(next(file))

formatted = tokenizer.apply_chat_template(
    example["messages"],
    tokenize=False,
)

print(formatted)
print("Character count:", len(formatted))
