# from peft import PeftModel
import copy
import json
from contextlib import nullcontext
from datetime import datetime

import datasets
import torch
from peft import (LoraConfig, PeftConfig, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CodeLlamaTokenizer, Trainer,
                          TrainerCallback, TrainingArguments,
                          default_data_collator)

from source.preprocess import load_repairllama_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Available device: {device}")

print("Loading the base model...")
# most lightweight model of CodeLlama for instruction prompt
base_model = "codellama/CodeLlama-7b-Instruct-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else "cpu",
    force_download=True,
    trust_remote_code=True,
    load_in_8bit=True,
    # resume_download=True,
    # local_files_only=True,
    # cache_dir="model/codellamma",
)

# # Tokenization
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    # resume_download=True,
    force_download=True,
    trust_remote_code=True,
    # cache_dir="model/codellamma",
)

model.config.use_cache = False

# tokenizer.add_eos_token = True
# tokenizer.pad_token_id = 0
# tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt_codellama(data_point):
    full_prompt = f"""You are a powerful code-fixing model. 
    Your job is to analyze and fix vulnerabilities in code. 
    You are given a snippet of vulnerable code and its context.

You must output the fixed version of the code snippet.

### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:
{data_point["answer"]}
"""
    return tokenize(full_prompt)


def generate_eval_prompt_codellama(data_point):
    full_prompt = f"""You are a powerful code-fixing model. 
    Your job is to analyze and fix vulnerabilities in code. 
    You are given a snippet of vulnerable code and its context.

You must output the fixed version of the code snippet.

### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:
"""
    return full_prompt


dataset = load_repairllama_dataset()

debug = False  # set to False to process the entire dataset
train_dataset = dataset["train"]
val_dataset = dataset["test"]

if debug:
    train_dataset = train_dataset.shuffle(seed=42).select(range(200))
    val_dataset = val_dataset.shuffle(seed=42).select(range(100))

tokenized_train_dataset = train_dataset.map(
    generate_and_tokenize_prompt_codellama)
tokenized_val_dataset = train_dataset.map(
    generate_and_tokenize_prompt_codellama)

# # Check the model performance
eval_sample = dataset["test"][1]

print("Evaluating the base model...")


def evaluate_model(model, tokenizer, eval_sample):
    eval_prompt = generate_eval_prompt_codellama(eval_sample)
    model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = tokenizer.decode(
            model.generate(
                **model_input, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id
            )[0],
            skip_special_tokens=True,
        )
    print(output)


evaluate_model(model, tokenizer, eval_sample)


def create_peft_config(model):
    """ Create a PEFT configuration"""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


model, lora_config = create_peft_config(model)

# configure wandb logging if you want to use it
# wandb_project = "patchT5"
# if len(wandb_project) > 0:
#     os.environ["WANDB_PROJECT"] = wandb_project

# if torch.cuda.device_count() > 1:
#     model.is_parallelizable = True
#     model.model_parallel = True


# 6. Training arguments
num_train_epochs = 100
batch_size = 16  # 128
per_device_train_batch_size = 8  # 32
gradient_accumulation_steps = batch_size // per_device_train_batch_size

output_dir = "PatchLlama-" + str(num_train_epochs)


training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    group_by_length=True,
    logging_strategy="steps",
    save_strategy="no",
    gradient_checkpointing=False,)
# training_args = TrainingArguments(
#     per_device_train_batch_size=per_device_train_batch_size,
#     gradient_accumulation_steps=gradient_accumulation_steps,
#     num_train_epochs=2,
#     warmup_steps=10,
#     max_steps=20,
#     learning_rate=2e-4,
#     fp16=True,
#     bf16=Fal
#     logging_steps=10,
#     optim="paged_adamw_32bit",
#     evaluation_strategy="steps",  # if val_set_size > 0 else "no",
#     save_strategy="no",
#     eval_steps=10,
#     save_steps=0,
#     output_dir=output_dir,
#     # load_best_model_at_end=True,
#     # ddp_find_unused_parameters=False if ddp else None,
#     # group sequences of roughly the same length together to speed up training
#     group_by_length=True,
#     gradient_checkpointing=False,
#     report_to="none",  # "wandb",  # if use_wandb else "none",
#     # run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
# )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=default_data_collator,
    # data_collator=DataCollatorForSeq2Seq(
    #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    # ),
)
# old_state_dict = model.state_dict
# model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
#     model, type(model)
# )

# Train and save the model
trainer.train()

trainer.model.save_pretrained(output_dir)
print("Saved the model to:", output_dir)

# 8. Evaluate the model
print("Evaluating the fine-tuned model...")
evaluate_model(model, tokenizer, eval_sample)

print("=" * 50)
print("Experiment Complete!")
print("=" * 50)
