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


class CodeLlamaModel:
    dash_line = "=" * 50

    def __init__(self, config, log):
        """ Initialize the CodeLlama model"""
        self.log = log
        self.config = config
        self.device = config["device"]

    def load_codellama_model(self):
        """ Load the CodeLlama model"""
        base_model = self.config["base_model"]
        self.log.info("Loading the CodeLLama base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            force_download=True,
            trust_remote_code=True,
            load_in_8bit=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            force_download=True,
            trust_remote_code=True,
        )

        model.config.use_cache = False

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.log.info("Model and tokenizer loaded successfully!")
        self.log.info(self.dash_line)
        return model, tokenizer

    def split_train_val_tokenize(self, dataset, tokenizer, debug=False):
        """ Split the dataset into train, validation and test sets"""
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]

        if debug:
            train_dataset = train_dataset.shuffle(seed=42).select(range(200))
            val_dataset = val_dataset.shuffle(seed=42).select(range(100))

        partial_generate_and_tokenize = partial(
            generate_and_tokenize_prompt_codellama, tokenizer=tokenizer)

        tokenized_train_dataset = train_dataset.map(
            partial_generate_and_tokenize)
        tokenized_val_dataset = train_dataset.map(
            partial_generate_and_tokenize)
        return tokenized_train_dataset, tokenized_val_dataset

    def evaluate_model_codellama(self, model, tokenizer, eval_sample):
        """ Evaluate the model"""
        self.log.info("Evaluating the base model...")
        eval_prompt = generate_eval_prompt_codellama(eval_sample)
        model_input = tokenizer(
            eval_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = tokenizer.decode(
                model.generate(
                    **model_input, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id
                )[0],
                skip_special_tokens=True,
            )
        self.log.info(output)
        self.log.info(f"\nHUMAN BASELINE: \n{eval_sample['answer']}\n")
        self.log.info(self.dash_line)

    def run_codellama(self):
        """ Run the CodeLlama model"""
        model, tokenizer = self.load_codellama_model()

        dataset = load_repairllama_dataset()

        tokenized_train_dataset, tokenized_val_dataset = self.split_train_val_tokenize(
            dataset, tokenizer, self.config["debug_mode"])

        # # Check the model performance
        eval_sample = dataset["test"][1]

        self.evaluate_model_codellama(model, tokenizer, eval_sample)

        model, lora_config = create_peft_config(model)

        run_id = "PatchLlama-" + \
            str(self.config['fine_tuning']['num_train_epochs']) + 'epoch-' + \
            datetime.now().strftime("%Y%m%d-%H%M%S")

        self.config['fine_tuning']['output_dir'] = os.path.join(
            os.path.join(self.config['fine_tuning']['output_dir'], run_id))

        trainer, instruct_model, tokenizer = fine_tune_codellama_model(
            self.config, model, tokenizer, tokenized_train_dataset, tokenized_val_dataset)

        self.log.info("Evaluating the fine-tuned model...")
        self.evaluate_model_codellama(instruct_model, tokenizer, eval_sample)

        # TODO: Implement the following methods
        # self.log.info("Generating test patches...")
        # results = eva.generate_fixes(
        #     model,
        #     instruct_model,
        #     tokenizer,
        #     dataset,
        #     result_csv,
        # )

        # self.log.info("Evaluating the models...")

        # eva.evaluate_rouge(results)

        # eva.evaluate_bleu(results)
