"""
Copyright (C) 2023 Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the Apachee-2.0 license- http://www.apache.org/licenses/

Project: PatchT5 - Code Language Models on Generating Vulnerability Security Fixes utilizing Commit Hunks
@Programmer: Guru Bhandari
"""
import os
from datetime import datetime
from functools import partial

import torch
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer, CodeLlamaTokenizer, BitsAndBytesConfig, GenerationConfig)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import source.utility as util
from source.finetune import (create_peft_config, fine_tune_codellama_model)
# from source.preprocess import load_repairllama_dataset, load_dataset_from_fixme
from source.prompt import (generate_and_tokenize_prompt_codellama,
                           generate_eval_prompt_codellama)
import source.evaluate as eva


class CodeLlamaModel:
    dash_line = "=" * 50

    def __init__(self, config, log):
        """ Initialize the CodeLlama model"""
        self.log = log
        self.config = config
        self.device = self.config["device"]

    def load_codellama_model(self):
        """ Load the CodeLlama model"""
        base_model = self.config["base_model"]
        use_4bit = self.config["use_4bit_quantization"]
        local_model_path = "models/CodeLLama-7b-quantized-4bit"

        # Check if the quantized model is already saved locally
        if os.path.exists(local_model_path):
            self.log.info("Loading the locally saved quantized model...")
            model = AutoModelForCausalLM.from_pretrained(local_model_path)
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        else:
            self.log.info("Quantized model not found locally.")
            self.log.info("Loading and quantizing the base model...")

            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                load_in_8bit_fp32_cpu_offload=not use_4bit
            )

            # Initialize the model with empty weights
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="auto" if self.device == "cuda" else "cpu",
                    quantization_config=quantization_config,
                )

            tokenizer = AutoTokenizer.from_pretrained(base_model)

            model.config.use_cache = False

            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            # Save the quantized model for future use
            self.log.info("Saving the quantized model for future use...")
            model.save_pretrained(local_model_path)
            tokenizer.save_pretrained(local_model_path)

        self.log.info("Model and tokenizer loaded successfully!")
        self.log.info(self.dash_line)
        return model, tokenizer

    def split_train_val_tokenize(self, dataset, tokenizer, debug=False):
        """ Split the dataset into train, validation and test sets"""
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]

        if debug:
            train_dataset = train_dataset.shuffle(seed=42).select(range(20))
            val_dataset = val_dataset.shuffle(seed=42).select(range(10))

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
        input_ids = tokenizer(
            eval_prompt, return_tensors="pt").input_ids.to(self.device)
        # Generate text
        model_output = model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(
                max_new_tokens=512,
                do_sample=True,  # sampling instead of greedy decoding
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            ),
        )
        output = tokenizer.decode(
            model_output[0], skip_special_tokens=True
        )
        self.log.info(output)
        self.log.info(f"\nHUMAN BASELINE: \n{eval_sample['fix']}\n")
        self.log.info(self.dash_line)

    def run_codellama(self, dataset):
        """ Run the CodeLlama model"""
        model, tokenizer = self.load_codellama_model()

        tokenized_train_dataset, tokenized_val_dataset = self.split_train_val_tokenize(
            dataset, tokenizer, self.config["debug_mode"])

        # # Check the model performance
        eval_sample = dataset["test"][1]

        self.evaluate_model_codellama(model, tokenizer, eval_sample)

        model, lora_config = create_peft_config(model)

        if self.config['only_compare']:
            self.log.info('=' * 50)
            self.log.info(
                "Comparing already fine-tuned with the base model...")

            instruct_model = AutoModelForCausalLM.from_pretrained(
                self.config["instruct_model"])

            tokenizer = CodeLlamaTokenizer.from_pretrained(
                self.config["instruct_model"])
        else:
            trainer, instruct_model, tokenizer = fine_tune_codellama_model(
                self.config, model, tokenizer, tokenized_train_dataset, tokenized_val_dataset)

        self.log.info("Evaluating the fine-tuned model...")
        self.evaluate_model_codellama(instruct_model, tokenizer, eval_sample)

        return model, instruct_model, tokenizer
