"""
Copyright (C) 2023 Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the Apachee-2.0 license- http://www.apache.org/licenses/

Project: PatchT5 - Code Language Models on Generating Vulnerability Security Fixes utilizing Commit Hunks
@Programmer: Guru Bhandari
"""

import os
import torch
from datetime import datetime
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    RobertaTokenizer,
)

# custom functions
from source.preprocess import load_dataset_from_fixme
from source.finetune import fine_tune_codet5_model
from source.prompt import prompt_fix
import source.evaluate as eva
import source.utility as util


class CodeT5Model:
    dash_line = "=" * 50

    def __init__(self, config, log):
        self.log = log
        self.config = config
        self.device = config["device"]

    def load_codet5_model(self):
        """ Load the CodeT5 model,
        Load the https://huggingface.co/Salesforce/codet5-base
        """
        model_name = self.config["base_model"]
        self.log.info("Loading the CodeT5 model...")
        if self.config["generation"]["tokenizer"] == "roberta":
            tokenizer = RobertaTokenizer.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device == "cpu" else torch.float32,
                trust_remote_code=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device == "cpu" else torch.float32,
                use_fast=True
            )
        self.log.info("Tokenizer loaded successfully!")

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True).to(self.device)

        # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
        # A decoder-only architecture is being used, but right-padding was detected!
        # For correct generation results, please set `padding_side='left'`
        # when initializing the tokenizer.
        # Set padding to 'left'
        tokenizer.padding_side = "left"

        self.log.info("Model loaded successfully!")
        self.log.info(f"Original Model: {model_name}")
        self.log.info(self.dash_line)
        return model, tokenizer

    def evaluate_model(self, model, tokenizer, dataset):
        """Evaluate the model on the dataset"""
        self.log.debug("Test the Model generating a simple code snippet")
        text = "def greet(user): print(f'hello <extra_id_0>!')"
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)

        generated_ids = model.generate(input_ids, max_length=256)

        gen_output = tokenizer.decode(
            generated_ids[0], skip_special_tokens=True)
        self.log.debug(f"Model generated output: {gen_output}")
        self.log.info(self.dash_line)

    def generate_prompt_fixes_on_shots(self, dataset, tokenizer, model):
        """Generate patches for the given examples"""

        self.log.info(self.dash_line)
        self.log.info("Generate Patch without Prompt Engineering")
        self.log.info(self.dash_line)

        example_index_to_fix = 1
        example_indices_full = [2]
        example_indices_full = [3, 4, 5]

        generation_config = GenerationConfig(
            max_new_tokens=self.config["generation"]["max_new_tokens"],
            do_sample=self.config["generation"]["do_sample"],
            temperature=self.config["generation"]["temperature"],
        )

        prompt_fix(dataset, tokenizer, model,
                   gen_config=None,
                   shot_type=None,
                   example_indices=None,
                   example_index_to_fix=example_index_to_fix,
                   )

        prompt_fix(dataset, tokenizer, model,
                   gen_config=None,
                   shot_type="zero",
                   example_indices=None,
                   example_index_to_fix=example_index_to_fix,
                   )

        # prompt_fix(dataset, tokenizer, model,
        #            gen_config=None,
        #            shot_type="few",
        #            example_indices=example_indices_full,
        #            example_index_to_fix=example_index_to_fix,
        #            )

        # prompt_fix(dataset, tokenizer, model,
        #            gen_config=generation_config,
        #            shot_type="few",
        #            example_indices=example_indices_full,
        #            example_index_to_fix=example_index_to_fix,
        #            )

        # prompt_fix(dataset, tokenizer, model,
        #            gen_config=generation_config,
        #            shot_type="zero",
        #            example_indices=example_indices_full,
        #            example_index_to_fix=example_index_to_fix,
        #            )

    def run_codet5(self, dataset):
        """ Run the CodeT5 model"""
        model, tokenizer = self.load_codet5_model()

        self.evaluate_model(model, tokenizer, dataset)
        self.generate_prompt_fixes_on_shots(dataset, tokenizer, model)
        self.log.info(eva.get_trainable_model_pars(model))

        fine_tune_codet5_model(dataset, model, tokenizer,
                               self.config['fine_tuning']['output_dir'])

        self.log.info(self.dash_line)
        self.log.info("Loading the fine-tuned model...")
        instruct_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config['fine_tuning']['output_dir'],
            torch_dtype=torch.bfloat16 if self.device == "cpu" else torch.float32,
        ).to(self.device)

        eva.show_original_instruct_fix(
            dataset, tokenizer, model, instruct_model, index=1)
        return model, instruct_model, tokenizer
