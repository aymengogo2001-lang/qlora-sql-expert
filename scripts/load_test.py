import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_Config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
)

print("Loading tokernizer")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("loading model in 4-bit")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config = bnb_Config,
    device_map = "auto",
)

mem = torch.cuda.memory_allocated() / 1e9

print(f"Model loaded. GPU memory used by tensors: {mem:.2f} GB" )
print(model)
print("Is 4-bit:", model.is_loaded_in_4bit)
