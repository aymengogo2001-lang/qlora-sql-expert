"""
QLoRA fine-tuning of Qwen2.5-1.5B-Instruct into a SQL expert.
Tuned to fit an RTX 3060 6GB:
  - 4-bit NF4 quantization (bitsandbytes)
  - LoRA adapters only (base model frozen)
  - batch size 1 + gradient accumulation
  - gradient checkpointing, paged 8-bit optimizer

Run:
  python scripts/train_qlora.py
"""
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "full-training")
MAX_SEQ_LEN = 1024


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files=os.path.join(DATA_DIR, "train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=os.path.join(DATA_DIR, "val.jsonl"), split="train")

    def formatting_func(batch):
        return [
            tokenizer.apply_chat_template(messages, tokenize=False)
            for messages in batch["messages"]
        ]
    
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        optim="paged_adamw_8bit",
        bf16=True,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
