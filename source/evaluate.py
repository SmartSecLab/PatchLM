# ### 2.4 - Evaluate the Model Quantitatively (with ROUGE Metric)
import numpy as np
import pandas as pd
import evaluate
from transformers import GenerationConfig
from codebleu import calc_codebleu
from tabulate import tabulate

# custom imports
from source.prompt import zero_prompt
import source.utility as util

# Setup logger
log = util.get_logger()
config = util.load_config()

rouge = evaluate.load("rouge")
dash_line = "=" * 50


def get_trainable_model_pars(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    percentage = 100 * trainable_model_params / all_model_params

    return (
        f"Trainable model parameters: {trainable_model_params}\n"
        f"All model parameters: {all_model_params}\n"
        f"Percentage of trainable model parameters: {percentage:.2f}%"
    )


def show_original_instruct_fix(
    dataset, tokenizer, original_model, instruct_model, index=2
):
    prompt = zero_prompt(dataset, index=index)
    human_baseline_fix = dataset["test"][index]["fix"]

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    original_model_outputs = original_model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            max_new_tokens=config["generation"]["max_new_tokens"],
            num_beams=config["generation"]["num_beams"],
        ),
    )
    original_model_text_output = tokenizer.decode(
        original_model_outputs[0], skip_special_tokens=True
    )

    instruct_model_outputs = instruct_model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            max_new_tokens=config["generation"]["max_new_tokens"],
            num_beams=config["generation"]["num_beams"],
        ),
    )
    instruct_model_text_output = tokenizer.decode(
        instruct_model_outputs[0], skip_special_tokens=True
    )

    log.info(dash_line)
    log.info(f"BASELINE PATCH:\n{human_baseline_fix}")
    log.info(dash_line)
    log.info(f"ORIGINAL MODEL:\n{original_model_text_output}")
    log.info(dash_line)
    log.info(f"INSTRUCT MODEL:\n{instruct_model_text_output}")


def generate_fixes(
    original_model,
    instruct_model,
    tokenizer,
    dataset,
    result_csv,
):
    """" Generate fixes for a list of vulnerables using a model """
    original_model_fixes = []
    instruct_model_fixes = []

    if config["debug_mode"]:
        vulnerables = dataset["test"][:4]["vulnerable"]
        human_baseline_fixes = dataset["test"][:4]["fix"]
        programming_languages = dataset["test"][:4]["programming_language"]
    else:
        vulnerables = dataset["test"]["vulnerable"]
        human_baseline_fixes = dataset["test"]["fix"]
        programming_languages = dataset["test"]["programming_language"]

    for _, vulnerable in enumerate(vulnerables):
        prompt = f"""
                    Generation the fix for the following vulnerable code:

                    {vulnerable}

                    fix: \n"""
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        original_model_outputs = original_model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(
                max_new_tokens=config["generation"]["max_new_tokens"],
            ),
        )
        original_model_text_output = tokenizer.decode(
            original_model_outputs[0], skip_special_tokens=True
        )

        original_model_fixes.append(original_model_text_output)

        instruct_model_outputs = instruct_model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(
                max_new_tokens=config["generation"]["max_new_tokens"],
            ),
        )
        instruct_model_text_output = tokenizer.decode(
            instruct_model_outputs[0], skip_special_tokens=True
        )

        instruct_model_fixes.append(instruct_model_text_output)

    zipped_fixes = list(
        zip(
            human_baseline_fixes,
            original_model_fixes,
            instruct_model_fixes,
            programming_languages,
        )
    )

    df = pd.DataFrame(
        zipped_fixes,
        columns=[
            "human_baseline_fixes",
            "original_model_fixes",
            "instruct_model_fixes",
            "programming_language",
        ],
    )
    df.to_csv(result_csv, index=False)
    log.info(dash_line)
    log.info(f"Results of vul-fix-training saved to {result_csv}")
    log.info(dash_line)
    log.info("Sample of the results:")
    log.info(df.head())
    log.info(dash_line)
    return df


def show_rouge_scores(original_model_results, instruct_model_results):
    df = pd.DataFrame(
        [original_model_results, instruct_model_results], index=["Base", "Instruct"]
    ).T
    df["Base"] = df["Base"] * 100
    df["Instruct"] = df["Instruct"] * 100
    df["Improvement"] = df["Instruct"] - df["Base"]
    df = df.round(2).applymap(lambda x: f"{x:.2f}%")
    log.info(
        f"The ROUGE scores improved: \n{
            tabulate(df, headers='keys', tablefmt='psql')}"
    )


def evaluate_rouge(results):
    """ Evaluate the fixes generated by the models using the ROUGE metric """
    human_baseline_fixes = results["human_baseline_fixes"].values
    original_model_fixes = results["original_model_fixes"].values
    instruct_model_fixes = results["instruct_model_fixes"].values

    original_model_results = rouge.compute(
        predictions=original_model_fixes,
        references=human_baseline_fixes[0: len(original_model_fixes)],
        use_aggregator=True,
        use_stemmer=True,
    )

    instruct_model_results = rouge.compute(
        predictions=instruct_model_fixes,
        references=human_baseline_fixes[0: len(instruct_model_fixes)],
        use_aggregator=True,
        use_stemmer=True,
    )

    log.info("ORIGINAL MODEL:")
    log.info(original_model_results)
    log.info("INSTRUCT MODEL:")
    log.info(instruct_model_results)

    log.info("Absolute percentage improvement of INSTRUCT over ORIGINAL MODEL")
    show_rouge_scores(original_model_results, instruct_model_results)
    log.info(dash_line)


def calc_codebleu_scores(
    references, predictions, langs, weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None
):
    """
    Calculate the CodeBLEU scores.

    Args:
        references (List[str]): List of reference code strings.
        predictions (List[str]): List of predicted code strings.
        langs (List[str]): List of programming languages for each pair.
        weights (Tuple[float]): Weights for 1-gram, 2-gram, 3-gram, 4-gram. Default is (0.25, 0.25, 0.25, 0.25).
        tokenizer (Optional): Tokenizer to use for tokenization. Default is None.

    Returns:
        List[Dict[str, float]]: List of dictionaries, each containing the CodeBLEU score and its components for a
        corresponding pair of reference and prediction.
    """
    scores = [
        calc_codebleu([ref], [pred], lang=lg,
                      weights=weights, tokenizer=tokenizer)
        for ref, pred, lg in zip(references, predictions, langs)
    ]
    return scores


def show_bleu_scores(original_bleu_scores, instruct_bleu_scores):
    """
    Display the improvement in BLEU scores.

    Args:
        original_bleu_scores (List[Dict[str, float]]): BLEU scores for the original model.
        instruct_bleu_scores (List[Dict[str, float]]): BLEU scores for the instruct model.
    """
    df_original = pd.DataFrame(original_bleu_scores).mean()
    df_instruct = pd.DataFrame(instruct_bleu_scores).mean()

    df = pd.concat([df_original, df_instruct], axis=1)
    df.columns = ["Base", "Instruct"]
    df["Base"] *= 100
    df["Instruct"] *= 100
    df["Improvement"] = df["Instruct"] - df["Base"]
    df = df.round(2).applymap(lambda x: f"{x:.2f}%")

    log.info(
        f"Weighted average BLEU scores improved: \n{
            tabulate(df, headers='keys', tablefmt='psql')}"
    )


def evaluate_bleu(results):
    """ Evaluate the fixes generated by the models using the CodeBLEU metric """
    try:
        human_baseline_fixes = results["human_baseline_fixes"].tolist()
        original_model_fixes = results["original_model_fixes"].tolist()
        instruct_model_fixes = results["instruct_model_fixes"].tolist()
        # guessland requires lower case the pl and replace 'c++' with 'cpp'
        langs = (
            results["programming_language"].str.lower().replace("c++",
                                                                "cpp").tolist()
        )

        original_bleu_scores = calc_codebleu_scores(
            references=human_baseline_fixes,
            predictions=original_model_fixes,
            langs=langs,
        )

        instruct_bleu_scores = calc_codebleu_scores(
            references=human_baseline_fixes,
            predictions=instruct_model_fixes,
            langs=langs,
        )

        show_bleu_scores(original_bleu_scores, instruct_bleu_scores)
    except Exception as e:
        log.warning(f"An error occurred evaluating bleu: {str(e)}")
