# QLoRA SQL Expert

A hands-on fine-tuning project that adapts **Qwen2.5-1.5B-Instruct** for text-to-SQL: given a database schema and a natural-language question, generate a SQL query. The model was fine-tuned with **4-bit QLoRA** on an NVIDIA RTX 3060 with 6 GB VRAM.

This repository contains the training and evaluation code. Generated dataset files, the Python environment, and model adapter weights are excluded from Git. Recreate the dataset and adapter locally using the commands below; the adapter is saved to `outputs/full-training`.

## Results

On a 200-example sample from the held-out test split, using greedy generation and normalized exact-string matching:

| Model | Exact match |
|---|---:|
| Qwen2.5-1.5B-Instruct (base) | 5% (10/200) |
| Qwen2.5-1.5B-Instruct + QLoRA adapter | 77% (154/200) |

Normalization removes Markdown code fences, ignores case and whitespace, and strips trailing semicolons. These are **exact-string match results**, not execution accuracy: logically equivalent SQL can be marked incorrect if written differently, and a matching string does not guarantee correctness for every database. The reported scores are a preliminary evaluation of the first 200 examples in the held-out split; they should not be interpreted as a full-test-set benchmark or a guarantee of correctness.

Training used 77,577 training examples for 2 epochs, with a 500-example validation split. The saved training state reports 9,696 optimizer steps and final validation loss of 0.3836. The adapter has approximately 18.46 million trainable parameters (about 1.18% of the 1.56B total parameters).

## Hardware and method

- GPU: NVIDIA GeForce RTX 3060, 6 GB VRAM
- Base model: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- Dataset: [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context)
- Quantization: 4-bit NF4 with double quantization
- Adapters: LoRA rank 16, alpha 32, dropout 0.05; attention and MLP projections
- Per-device batch size: 1; gradient accumulation: 16
- Sequence limit: 1,024 tokens; epochs: 2
- Optimizer: paged AdamW 8-bit; gradient checkpointing enabled

The dataset was shuffled and split by example, not grouped by schema. Related schemas or patterns may occur across splits, so the evaluation should be interpreted as a project benchmark, not a guarantee of generalization to unseen real-world databases.

## Repository layout

```text
.
├── requirements.txt
├── scripts/
│   ├── prepare_dataset.py  # Download and format the public dataset
│   ├── train_qlora.py      # QLoRA supervised fine-tuning
│   ├── eval.py             # Base-vs-adapter exact-match evaluation
│   ├── chat.py             # Interactive schema/question prompt
│   ├── load_test.py        # 4-bit model loading exercise
│   ├── generate_base.py    # Base-model generation exercise
│   └── preview_format.py   # Inspect the Qwen chat template
└── outputs/                # Ignored: local adapters/checkpoints
```

The `data/*.jsonl` files and `venv/` are generated/local files and are ignored by Git.

## Setup (Windows PowerShell)

```powershell
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt
```

The requirements pin PyTorch 2.3.1 with CUDA 12.1 wheels and Transformers 4.46.x for compatibility with the training code. A compatible NVIDIA driver and a supported Windows/Python setup are required for GPU training; check the PyTorch and bitsandbytes support matrix if adapting the environment.

## Reproduce

From the project directory:

```powershell
# Download and create train/validation/test JSONL files
venv/Scripts/python scripts/prepare_dataset.py

# Train (requires the dataset files and the stated GPU setup)
venv/Scripts/python scripts/train_qlora.py

# Compare base model and locally trained adapter on the 200-example sample
venv/Scripts/python scripts/eval.py --n 200

# Evaluate all 500 examples in the held-out split
venv/Scripts/python scripts/eval.py --n 500

# Interactively generate SQL from a schema and question
venv/Scripts/python scripts/chat.py
```

Both scripts default to the adapter saved by `train_qlora.py` at `outputs/full-training`. To use an adapter stored elsewhere, pass its local folder with `--adapter PATH`.

## Responsible use

Generated SQL can be wrong, costly, or unsafe. The interactive script only generates text; it does not connect to or execute against a database. Before connecting this model to a database, use a dedicated least-privilege read-only account, validate generated statements, enforce row/time limits, and require review before execution. Never expose database credentials in source code or commit them to Git.

## License and attribution

No license is granted for this repository's code or adapter weights. The base model and dataset are third-party materials governed by their own terms; consult the [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and the [dataset card](https://huggingface.co/datasets/b-mc2/sql-create-context) before reuse or redistribution. The model and dataset are cited above.
