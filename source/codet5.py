# Generative AI Use Case: Patches Generation
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

dash_line = "=" * 50

# Setup logger
log = util.get_logger()
config = util.load_config()
log.info(dash_line)
log.info(f"Logging  at: {util.log_filename}")
log.info(f"Config: {config}")
log.info(dash_line)


def load_codet5_model(config):
    """ Load the CodeT5 model,
    Load the https://huggingface.co/Salesforce/codet5-base
    """
    model_name = config["base_model"]
    log.info("Loading the CodeT5 model...")
    if config["generation"]["tokenizer"] == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    log.info("Tokenizer loaded successfully!")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, trust_remote_code=True)
    log.info("Model loaded successfully!")
    log.info(f"Original Model: {model_name}")
    log.info(dash_line)
    return model, tokenizer


def evaluate_model(model, tokenizer, dataset):
    """Evaluate the model on the dataset"""
    log.debug("Test the Model generating a simple code snippet")
    text = "def greet(user): print(f'hello <extra_id_0>!')"
    # text = "def add(a, b): \n int sum= a + b \n return sum"
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    # simply generate a single sequence
    generated_ids = model.generate(input_ids, max_length=256)
    gen_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    log.debug(f"Model generated output: {gen_output}")
    # this prints "{user.username}"
    log.info(dash_line)


def generate_prompt_fixes_on_shots(dataset, tokenizer, model):
    """Generate patches for the given examples"""
    # Now it's time to explore how well the base LLM fixs a vulnerable
    # without any prompt engineering. **Prompt engineering** is an act
    # of a human changing the **prompt** (input) to improve the response
    # for a given task.
    example_index_to_fix = 2

    # ### 2.1 - without Prompt Engineering
    log.info(dash_line)
    log.info("Generate Patch without Prompt Engineering")
    log.info(dash_line)

    prompt_fix(
        dataset,
        tokenizer,
        model,
        gen_config=None,
        shot_type=None,
        example_indices=None,
        example_index_to_fix=example_index_to_fix,
    )

    # 3 - fix vulnerable with an Instruction Prompt
    # ### 3.1 - Zero Shot Inference with an Instruction Prompt

    prompt_fix(
        dataset,
        tokenizer,
        model,
        gen_config=None,
        shot_type="zero",
        example_indices=None,
        example_index_to_fix=example_index_to_fix,
    )

    # ## 4 - fix vulnerable with One Shot and Few Shot Inference
    # ### 4.1 - One Shot Inference

    example_indices_full = [2]
    example_index_to_fix = 3

    prompt_fix(
        dataset,
        tokenizer,
        model,
        gen_config=None,
        shot_type="few",
        example_indices=example_indices_full,
        example_index_to_fix=example_index_to_fix,
    )

    # ### 4.2 - Few Shot Inference

    example_indices_full = [2, 3, 4]
    example_index_to_fix = 1

    prompt_fix(
        dataset,
        tokenizer,
        model,
        gen_config=None,
        shot_type="few",
        example_indices=example_indices_full,
        example_index_to_fix=example_index_to_fix,
    )

    generation_config = GenerationConfig(
        max_new_tokens=config["generation"]["max_new_tokens"],
        do_sample=config["generation"]["do_sample"],
        temperature=config["generation"]["temperature"],
    )

    prompt_fix(
        dataset,
        tokenizer,
        model,
        gen_config=generation_config,
        shot_type="few",
        example_indices=example_indices_full,
        example_index_to_fix=example_index_to_fix,
    )

    # ### 1.2 - Get the Trainable Parameters of the Model
    log.info(eva.get_trainable_model_pars(model))

    # ### 1.3 - Test the Model with Zero Shot Inferencing
    prompt_fix(
        dataset,
        tokenizer,
        model,
        gen_config=generation_config,
        shot_type="zero",
        example_indices=example_indices_full,
        example_index_to_fix=example_index_to_fix,
    )


def run_codet5():
    # # ============= Load the model and tokenizer =============
    model, tokenizer = load_codet5_model(config)

    # # ============= Load the dataset =============
    dataset = load_dataset_from_df()

    # # ============= Test the Model ===========================

    evaluate_model(model, tokenizer, dataset)

    generate_prompt_fixes_on_shots(dataset, tokenizer, model)

    output_dir = f"models/instruct-model-{config['run_id']}"

    fine_tune_codet5_model(dataset, model, tokenizer, output_dir)

    # ### 2.2 - Load the fine-tuned Model
    log.info(dash_line)
    log.info("Loading the fine-tuned model...")
    instruct_model = AutoModelForSeq2SeqLM.from_pretrained(
        output_dir, torch_dtype=torch.bfloat16)

    # ### 2.3 - Evaluate the Model Qualitatively (Human Evaluation)
    eva.show_original_instruct_fix(
        dataset, tokenizer, model, instruct_model, index=1)

    # ### 2.4 - Evaluate the Model Quantitatively (ROUGE)

    result_csv = util.log_dir / f"result-{util.run_id}.csv"

    log.info("Generating test patches...")
    # generate patches for the test dataset
    results = eva.generate_fixes(
        model,
        instruct_model,
        tokenizer,
        dataset,
        result_csv,
    )

    log.info("Evaluating the models...")

    eva.evaluate_rouge(results)

    eva.evaluate_bleu(results)

    log.info(dash_line)
    log.info("End of the run")
    log.info(dash_line)
