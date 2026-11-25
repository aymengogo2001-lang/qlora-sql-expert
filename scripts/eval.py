"""
Compare BASE Qwen2.5-1.5B-Instruct vs the QLoRA fine-tuned SQL expert
on the held-out test set (data/test.jsonl).

Metric: exact-match (normalized whitespace/case) against the reference
SQL query. Also prints a few side-by-side samples so you can show
"before vs after" in a README/portfolio.

Run:
  python scripts/eval.py --adapter outputs/qlora-sql-expert --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional

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


def normalize(sql: str) -> str:
    sql = re.sub(r"^\s*```(?:sql)?\s*|\s*```\s*$", "", sql, flags=re.IGNORECASE)
    sql = sql.strip().strip(";").lower()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def load_model(adapter_path: str | None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, context: str, question: str) -> str:
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
    return text.strip()


def run_eval(model, tokenizer, rows, label: str):
    correct = 0
    samples = []
    for row in rows:
        pred = generate(model, tokenizer, row["context"], row["question"])
        gold = row["answer"]
        is_correct = normalize(pred) == normalize(gold)
        correct += int(is_correct)
        samples.append((row["question"], gold, pred, is_correct))
    acc = correct / len(rows) if rows else 0.0
    print(f"\n=== {label}: exact-match accuracy = {acc:.2%} ({correct}/{len(rows)}) ===")
    for q, gold, pred, ok in samples[:3]:
        print(f"Q: {q}\n  gold: {gold}\n  pred: {pred}\n  match: {ok}\n")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/qlora-sql-expert")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--skip-base", action="store_true", help="only eval fine-tuned model")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    test_path = os.path.join(data_dir, "test.jsonl")
    rows = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows = rows[: args.n]

    if not args.skip_base:
        print("Loading BASE model (no fine-tuning) ...")
        base_model, base_tok = load_model(None)
        run_eval(base_model, base_tok, rows, "BASE Qwen2.5-1.5B-Instruct")
        del base_model
        torch.cuda.empty_cache()

    print("Loading FINE-TUNED model (QLoRA adapter) ...")
    ft_model, ft_tok = load_model(args.adapter)
    run_eval(ft_model, ft_tok, rows, "Fine-tuned SQL expert")


if __name__ == "__main__":
    main()
