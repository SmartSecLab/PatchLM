# Generative AI Use Case: Patches Generation

from transformers import GenerationConfig

# custom functions
import source.utility as util


# Setup logger
log = util.get_logger()
config = util.load_config()

dash_line = "-" * 20


def show_few_examples(dataset, num_examples=2):
    # Print the first few vulnerables and summaries
    log.info("Examples in the dataset:")
    example_indices = [2, 4]
    # dash_line = '=' * 50

    for i, index in enumerate(example_indices):
        log.info(dash_line)
        log.info(f"Example {i + 1}")
        log.info(dash_line)
        log.info("Vulnerable code:")
        log.info(dataset["test"][index]["vulnerable"])
        log.info(dash_line)
        log.info("BASELINE PATCH:")
        log.info(dataset["test"][index]["fix"])
        log.info(dash_line)
        log.info()


# -


def zero_prompt(dataset, index=2):
    vulnerable = dataset["test"][index]["vulnerable"]
    return f"""
    Vulnerable program code:

    {vulnerable}

    Patch of the program is:
    """


def one_few_prompt(dataset, example_indices, example_index_to_fix):
    """Construct the prompt to perform one shot inference:"""
    prompt = ""
    for index in example_indices:
        vulnerable = dataset["test"][index]["vulnerable"]
        fix = dataset["test"][index]["fix"]

        # The stop sequence '{fix}\n\n\n' is important for FLAN-T5. Other models may have their own preferred stop sequence.
        prompt += f"""
                    Vulnerable C program:

                    {vulnerable}

                    Patch of the program is:

                    {fix}

                    """

    vulnerable = dataset["test"][example_index_to_fix]["vulnerable"]

    prompt += f"""
                Vulnerable program code:

                {vulnerable}

                Patch of the program is:
                """

    return prompt


def without_prompt(dataset, index=2):
    vulnerable = dataset["test"][index]["vulnerable"]
    return vulnerable


def generate_fix(prompt, tokenizer, model, gen_config=None):
    """
    This line defines a function called generate_fix that takes four parameters:
    prompt: The text to be fixed.
    tokenizer: A tokenizer object that converts text into numerical tokens, usually to be processed by a model.
    model: A language model that generates text. This could be something like GPT (Generative Pre-trained Transformer) model.
    gen_config: Optional configurations for the text generation process.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    if gen_config is None:
        gen_config = GenerationConfig(max_length=200)

    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=config["generation"]["max_new_tokens"],
            generation_config=gen_config,
        )[0],
        skip_special_tokens=True,
    )
    return output


def prompt_fix(
    dataset,
    tokenizer,
    model,
    gen_config=None,
    shot_type="zero",
    example_indices=None,
    example_index_to_fix=2,
):
    dash_line = "=" * 25
    if shot_type == "zero":
        prompt = zero_prompt(dataset, example_index_to_fix)
    elif shot_type == "one_few":
        prompt = one_few_prompt(dataset, example_indices, example_index_to_fix)
    else:
        prompt = without_prompt(dataset, example_index_to_fix)

    prompt = zero_prompt(dataset)

    fix = dataset["test"][example_index_to_fix]["fix"]
    output = generate_fix(prompt, tokenizer, model, gen_config)

    dash_line = "-" * 100
    log.info(dash_line)
    log.info(f"INPUT PROMPT:\n{prompt}")
    log.info(dash_line)
    log.info(f"BASELINE PATCH:\n{fix}\n")
    log.info(dash_line)
    log.info(f"MODEL GENERATION - ZERO SHOT:\n{output}")


def tokenize(prompt, tokenizer):
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


def generate_and_tokenize_prompt_codellama(data_point, tokenizer):
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
    return tokenize(full_prompt, tokenizer)


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
