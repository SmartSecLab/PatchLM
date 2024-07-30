import os
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    RobertaTokenizer,
)

# custom functions
from source.preprocess import load_dataset_from_df
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
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True)
        self.log.info("Tokenizer loaded successfully!")

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True)
        self.log.info("Model loaded successfully!")
        self.log.info(f"Original Model: {model_name}")
        self.log.info(self.dash_line)
        return model, tokenizer

    def evaluate_model(self, model, tokenizer, dataset):
        """Evaluate the model on the dataset"""
        self.log.debug("Test the Model generating a simple code snippet")
        text = "def greet(user): print(f'hello <extra_id_0>!')"
        input_ids = tokenizer(text, return_tensors="pt").input_ids

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

        prompt_fix(dataset, tokenizer, model,
                   gen_config=None,
                   shot_type="few",
                   example_indices=example_indices_full,
                   example_index_to_fix=example_index_to_fix,
                   )

        prompt_fix(dataset, tokenizer, model,
                   gen_config=generation_config,
                   shot_type="few",
                   example_indices=example_indices_full,
                   example_index_to_fix=example_index_to_fix,
                   )

        prompt_fix(dataset, tokenizer, model,
                   gen_config=generation_config,
                   shot_type="zero",
                   example_indices=example_indices_full,
                   example_index_to_fix=example_index_to_fix,
                   )
        self.log.info(eva.get_trainable_model_pars(model))

    def run_codet5(self):
        dataset = load_dataset_from_df()
        model, tokenizer = self.load_codet5_model()

        self.evaluate_model(model, tokenizer, dataset)

        self.generate_prompt_fixes_on_shots(dataset, tokenizer, model)

        output_dir = f"models/instruct-model-{self.config['run_id']}"

        fine_tune_codet5_model(dataset, model, tokenizer, output_dir)

        self.log.info(self.dash_line)
        self.log.info("Loading the fine-tuned model...")
        instruct_model = AutoModelForSeq2SeqLM.from_pretrained(
            output_dir, torch_dtype=torch.bfloat16)

        eva.show_original_instruct_fix(
            dataset, tokenizer, model, instruct_model, index=1)

        result_csv = util.log_dir / f"result-{util.run_id}.csv"

        self.log.info("Generating test patches...")
        results = eva.generate_fixes(
            model,
            instruct_model,
            tokenizer,
            dataset,
            result_csv,
        )

        self.log.info("Evaluating the models...")

        eva.evaluate_rouge(results)

        eva.evaluate_bleu(results)
