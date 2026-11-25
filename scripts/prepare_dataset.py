"""
Download b-mc2/sql-create-context and turn it into a Qwen2.5 chat-format
SFT dataset: (schema + question) -> SQL query.

Produces:
  data/train.jsonl
  data/val.jsonl
  data/test.jsonl   (held out, used by eval.py)

Each line is a JSON object with a "messages" list in the standard
chat format, ready for the Qwen2.5 chat template.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

RANDOM_SEED = 42
VAL_SIZE = 500
TEST_SIZE = 500

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema (as CREATE TABLE "
    "statements) and a natural language question, write a single correct "
    "SQL query that answers the question. Reply with ONLY the SQL query, "
    "no explanation."
)


def to_chat_example(row: dict) -> dict:
    question = row["question"].strip()
    context = row["context"].strip()
    answer = row["answer"].strip()
    user_content = f"### Database schema:\n{context}\n\n### Question:\n{question}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading b-mc2/sql-create-context ...")
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    print(f"Total examples: {len(ds)}")

    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    test_idx = set(indices[:TEST_SIZE])
    val_idx = set(indices[TEST_SIZE:TEST_SIZE + VAL_SIZE])

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = out_dir / "test.jsonl"

    with train_path.open("w", encoding="utf-8") as f_train, \
         val_path.open("w", encoding="utf-8") as f_val, \
         test_path.open("w", encoding="utf-8") as f_test:
        for i, row in enumerate(ds):
            example = to_chat_example(row)
            # keep raw fields in test set for exact-match / execution-style eval
            if i in test_idx:
                example["question"] = row["question"]
                example["context"] = row["context"]
                example["answer"] = row["answer"]
                f_test.write(json.dumps(example) + "\n")
            elif i in val_idx:
                f_val.write(json.dumps(example) + "\n")
            else:
                f_train.write(json.dumps(example) + "\n")

    print(f"Wrote {train_path} / {val_path} / {test_path}")


if __name__ == "__main__":
    main()
