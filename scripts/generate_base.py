import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()

messages = [
    {"role": "system", "content": "You are a SQL expert. Given a database schema and a question, write one SQL query. Reply with only the SQL."},
    {"role": "user", "content": "### Database schema:\nCREATE TABLE employees (id INT, name TEXT, salary INT, dept TEXT);\n\n### Question:\nWhat is the average salary per department?"},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)

answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("BASE MODEL SQL:")
print(answer)
