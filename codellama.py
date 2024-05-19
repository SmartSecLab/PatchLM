from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# most lightweight model of CodeLlama for instruction prompt
model_id = "codellama/CodeLlama-7b-Instruct-hf"
# model_id = '/Users/guru/research/LLMs/CodeLlama-70-Instruct-hf'
tokenizer = AutoTokenizer.from_pretrained(model_id, force_download=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

chat = [
    {"role": "system", "content": "You are a helpful and honest code assistant expert in JavaScript. Please, provide all answers to programming questions in JavaScript"},
    {"role": "user", "content": "Write a function that computes the set of sums of all contiguous sublists of a given list."},
]
# inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")
inputs = tokenizer.apply_chat_template(chat, return_tensors="pt")

output = model.generate(input_ids=inputs, max_new_tokens=200)
output = output[0].to("cpu")
# model.save_pretrained("/Users/guru/research/LLMs/CodeLlama-7b-Instruct-hf-local", safe_serialization=False)
print(tokenizer.decode(output))

# +
# model.save_pretrained("/Users/guru/research/LLMs/CodeLlama-7b-Instruct-hf-local", safe_serialization=False)

# +
chat = [
    {"role": "system", "content": "You are a helpful and honest code assistant expert in JavaScript. Please, provide all answers to programming questions in JavaScript"},
    {"role": "user", "content": "Write a function that computes the set of sums of all contiguous sublists of a given list."},
]
# inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")
inputs = tokenizer.apply_chat_template(chat, return_tensors="pt")

output = model.generate(input_ids=inputs, max_new_tokens=200)
output = output[0].to("cpu")
print(tokenizer.decode(output))
