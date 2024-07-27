# from peft import PeftModel
import copy
import json
import os
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import datasets
import torch
from peft import (LoraConfig, PeftConfig, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from peft.peft_model import get_peft_model_state_dict
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CodeLlamaTokenizer, Trainer,
                          TrainerCallback, TrainingArguments,
                          default_data_collator)

import source.evaluate as eva
import source.utility as util
from source.finetune import fine_tune_model
from source.preprocess import load_repairllama_dataset
from source.prompt import (generate_and_tokenize_prompt_codellama,
                           generate_eval_prompt_codellama, prompt_fix)

dash_line = "=" * 50

# Setup logger
log = util.get_logger()
config = util.load_config()
log.info(dash_line)
log.info(f"Logging  at: {util.log_filename}")
log.info(f"Config: {config}")
log.info(dash_line)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using available device: {device}")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading the base model...")


def load_codellama_model(config):
    """ Load the CodeLlama model"""
    base_model = config["base_model"]
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

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        # resume_download=True,
        force_download=True,
        trust_remote_code=True,
        # cache_dir="model/codellamma",
    )

    model.config.use_cache = False

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


model, tokenizer = load_codellama_model(config)


dataset = load_repairllama_dataset()

debug = False  # set to False to process the entire dataset
train_dataset = dataset["train"]
val_dataset = dataset["test"]

if debug:
    train_dataset = train_dataset.shuffle(seed=42).select(range(200))
    val_dataset = val_dataset.shuffle(seed=42).select(range(100))


partial_generate_and_tokenize = partial(
    generate_and_tokenize_prompt_codellama, tokenizer=tokenizer)

tokenized_train_dataset = train_dataset.map(partial_generate_and_tokenize)
tokenized_val_dataset = train_dataset.map(partial_generate_and_tokenize)

# # Check the model performance
eval_sample = dataset["test"][1]

print("Evaluating the base model...")


def evaluate_model(model, tokenizer, eval_sample):
    eval_prompt = generate_eval_prompt_codellama(eval_sample)
    model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = tokenizer.decode(
            model.generate(
                **model_input, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id
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

if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True


# 6. Training arguments
num_train_epochs = config['fine_tuning']['num_train_epochs']
batch_size = config['fine_tuning']['batch_size']
per_device_train_batch_size = config['fine_tuning']['per_device_train_bsize']
gradient_accumulation_steps = batch_size // per_device_train_batch_size

output_dir = os.path.join(
    config['fine_tuning']['output_dir'], "PatchLlama-" + str(num_train_epochs))


training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim="paged_adamw_32bit",
    save_steps=5,
    logging_steps=5,
    learning_rate=config['fine_tuning']['learning_rate'],
    evaluation_strategy="steps",
    eval_steps=5,
    fp16=True,
    bf16=False,
    group_by_length=True,
    logging_strategy="steps",
    save_strategy="no",
    gradient_checkpointing=False,)


trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=default_data_collator,
)

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)

# Train and save the model
trainer.train()

trainer.model.save_pretrained(output_dir)
trainer.save_model(output_dir)
print("Saved the model to:", output_dir)

# 8. Evaluate the model
print("Evaluating the fine-tuned model...")
evaluate_model(model, tokenizer, eval_sample)

print("=" * 50)
print("Experiment Complete!")
print("=" * 50)
