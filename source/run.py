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
from source.finetune import (create_peft_config, fine_tune_codellama_model,
                             fine_tune_model)
from source.preprocess import load_repairllama_dataset
from source.prompt import (generate_and_tokenize_prompt_codellama,
                           generate_eval_prompt_codellama, prompt_fix)
from source.run_codellama import load_codellama_model
from source.run_codet5 import load_codet5_model


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

if __name__ == "__main__":
    # Load the CodeLlama model or the CodeT5 model
    if 'codellama' in str(config['base_model']).lower():
        model, tokenizer = load_codellama_model(config)

    elif 'codet5' in str(config['base_model']).lower():
        model, tokenizer = load_codet5_model(config)

    else:
        raise ValueError(f"Invalid model name: {config['base_model']}")
    log.info(dash_line)
    log.info(f"Model loaded successfully: {config['base_model']}")
    log.info(dash_line)
