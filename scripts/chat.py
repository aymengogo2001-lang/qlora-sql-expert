"""
Interactive test of the fine-tuned SQL expert.

Run:
  python scripts/chat.py --adapter outputs/qlora-sql-expert

Paste a schema (CREATE TABLE ...) then a question, e.g.:
  schema> CREATE TABLE employees (id INT, name TEXT, salary INT, dept TEXT)
  question> What is the average salary per department?
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema (as CREATE TABLE "
    "statements) and a natural language question, write a single correct "
    "SQL query that answers the question. Reply with ONLY the SQL query, "
    "no explanation."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/qlora-sql-expert")
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    print("Type 'quit' to exit.\n")
    while True:
        context = input("schema> ")
        if context.strip().lower() == "quit":
            break
        question = input("question> ")
        if question.strip().lower() == "quit":
            break

        user_content = f"### Database schema:\n{context}\n\n### Question:\n{question}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nSQL>\n{text.strip()}\n")


if __name__ == "__main__":
    main()
