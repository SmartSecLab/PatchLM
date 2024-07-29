import os
from datetime import datetime
from functools import partial

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer)

import source.utility as util
from source.finetune import (create_peft_config, fine_tune_codellama_model)
from source.preprocess import load_repairllama_dataset
from source.prompt import (generate_and_tokenize_prompt_codellama,
                           generate_eval_prompt_codellama)


dash_line = "=" * 50

# Setup logger
log = util.get_logger()
config = util.load_config()
log.info(dash_line)
log.info(f"Logging  at: {util.log_filename}")
log.info(f"Config: {config}")
log.info(dash_line)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using available device: {device}")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_codellama_model(config):
    """ Load the CodeLlama model"""
    base_model = config["base_model"]
    log.info("Loading the CodeLLama base model...")
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
    log.info("Model and tokenizer loaded successfully!")
    log.info(dash_line)
    return model, tokenizer


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


def evaluate_model_codellama(model, tokenizer, eval_sample):
    """ Evaluate the model"""
    log.info("Evaluating the base model...")
    eval_prompt = generate_eval_prompt_codellama(eval_sample)
    model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = tokenizer.decode(
            model.generate(
                **model_input, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id
            )[0],
            skip_special_tokens=True,
        )
    log.info(output)
    log.info(f"\nHUMAN BASELINE: \n{eval_sample['answer']}\n")
    log.info(dash_line)


def run_codellama():
    """ Run the CodeLlama model"""
    model, tokenizer = load_codellama_model(config)

    dataset = load_repairllama_dataset()

    tokenized_train_dataset, tokenized_val_dataset = split_train_val_tokenize(
        dataset, tokenizer, config["debug_mode"])

    # # Check the model performance
    eval_sample = dataset["test"][1]

    evaluate_model_codellama(model, tokenizer, eval_sample)

    model, lora_config = create_peft_config(model)

    run_id = "PatchLlama-" + \
        str(config['fine_tuning']['num_train_epochs']) + 'epoch-' + \
        datetime.now().strftime("%Y%m%d-%H%M%S")

    config['fine_tuning']['output_dir'] = os.path.join(
        os.path.join(config['fine_tuning']['output_dir'], run_id))

    trainer, instruct_model, tokenizer = fine_tune_codellama_model(
        config, model, tokenizer, tokenized_train_dataset, tokenized_val_dataset)

    log.info("Evaluating the fine-tuned model...")
    evaluate_model_codellama(instruct_model, tokenizer, eval_sample)


if __name__ == "__main__":
    run_codellama()
