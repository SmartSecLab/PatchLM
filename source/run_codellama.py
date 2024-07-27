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
from source.finetune import fine_tune_model, fine_tune_codellama_model, create_peft_config
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


def split_train_val_tokenize(dataset, tokenizer, debug=False):
    """ Split the dataset into train, validation and test sets"""
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    if debug:
        train_dataset = train_dataset.shuffle(seed=42).select(range(200))
        val_dataset = val_dataset.shuffle(seed=42).select(range(100))

    partial_generate_and_tokenize = partial(
        generate_and_tokenize_prompt_codellama, tokenizer=tokenizer)

    tokenized_train_dataset = train_dataset.map(partial_generate_and_tokenize)
    tokenized_val_dataset = train_dataset.map(partial_generate_and_tokenize)
    return tokenized_train_dataset, tokenized_val_dataset


def evaluate_model(model, tokenizer, eval_sample):
    """ Evaluate the model"""
    print("Evaluating the base model...")
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
    print(f"\nExpected: {eval_sample['fix']}")
    print(dash_line)


tokenized_train_dataset, tokenized_val_dataset = split_train_val_tokenize(
    dataset, tokenizer, config["debug_mode"])

# # Check the model performance
eval_sample = dataset["test"][1]


evaluate_model(model, tokenizer, eval_sample)

model, lora_config = create_peft_config(model)

output_dir = os.path.join(
    config['fine_tuning']['output_dir'], "PatchLlama-" + str(config['fine_tuning']['num_train_epochs']))


trainer, model, tokenizer = fine_tune_codellama_model(
    config, model, tokenizer, tokenized_train_dataset, tokenized_val_dataset, output_dir)

print("Evaluating the fine-tuned model...")
evaluate_model(model, tokenizer, eval_sample)
