# # Prompt Engineering using CodeLlamma API
#

# +
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import sys
import os
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# Assuming you're using CUDA, set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f"Available device: {device}")
# -

# # Load the dataset

# Load ir1xor1
dataset = load_dataset("ASSERT-KTH/repairllama-datasets", "ir1xor1")
# Load irXxorY
# dataset = load_dataset("ASSERT-KTH/repairllama-dataset", "irXxorY")


def add_question(example):
    """ Add a new feature- question to the dataset """
    if "question" not in example:
        example[
            "question"
        ] = "What is the fix version of the code for the following vulnerability?"
    return example


def prepare_examples(dataset):
    """ Similarize the dataset by adding a question to the dataset  and renaming the columns"""
    dataset = dataset.map(add_question)
    # rename the columns
    dataset = dataset.rename_column("input", "context")
    dataset = dataset.rename_column("output", "answer")
    return dataset


dataset = prepare_examples(dataset)
print(dataset)
# -

# most lightweight model of CodeLlama for instruction prompt
base_model = "codellama/CodeLlama-7b-Instruct-hf"
# base_model = '/Users/guru/research/LLMs/CodeLlama-70-Instruct-hf'
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # load_in_8bit=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# # Tokenization
tokenizer = AutoTokenizer.from_pretrained(base_model, force_download=True)

# +
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = f"""You are a powerful code-fixing model. Your job is to analyze and fix vulnerabilities in code. You are given a snippet of vulnerable code and its context.

You must output the fixed version of the code snippet.

### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:
{data_point["answer"]}
"""
    return tokenize(full_prompt)


def generate_eval_prompt(data_point):
    full_prompt = f"""You are a powerful code-fixing model. Your job is to analyze and fix vulnerabilities in code. You are given a snippet of vulnerable code and its context.

You must output the fixed version of the code snippet.

### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:
"""
    return full_prompt


# -

# Reformat to prompt and tokenize each sample:
debug = True  # set to False to process the entire dataset
train_dataset = dataset['train']
val_dataset = dataset['test']

if debug:
    train_dataset = train_dataset.shuffle(seed=42).select(range(20))
    val_dataset = val_dataset.shuffle(seed=42).select(range(10))

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = train_dataset.map(generate_and_tokenize_prompt)

# # Check the model performance
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_prompt = generate_eval_prompt(dataset["test"][0])

model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input,
          max_new_tokens=200)[0], skip_special_tokens=True))


# # Training
# +
model.train()  # put model back into training mode
# model = prepare_model_for_int8_training(model)
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)


# set this to the adapter_model.bin file you want to resume from
resume_from_checkpoint = ""

if resume_from_checkpoint:
    if os.path.exists(resume_from_checkpoint):
        print(f"Restarting from {resume_from_checkpoint}")
        adapters_weights = torch.load(resume_from_checkpoint)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {resume_from_checkpoint} not found")

# configure wandb logging if you want to use it
# wandb_project = "patchT5"
# if len(wandb_project) > 0:
#     os.environ["WANDB_PROJECT"] = wandb_project

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True


# 6. Training arguments
batch_size = 128
per_device_train_batch_size = 32
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "patch-code-llama"

training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=100,
    max_steps=400,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",  # if val_set_size > 0 else "no",
    save_strategy="steps",
    eval_steps=20,
    save_steps=20,
    output_dir=output_dir,
    # save_total_limit=3,
    load_best_model_at_end=False,
    # ddp_find_unused_parameters=False if ddp else None,
    # group sequences of roughly the same length together to speed up training
    group_by_length=True,
    report_to="none",  # "wandb",  # if use_wandb else "none",
    # if use_wandb else None,
    run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

# Pytorch-related optimization to make it faster
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)

trainer.train()


# 7. Load the final checkpoint

# base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# To load a fine-tuned Lora/Qlora adapter use PeftModel.from_pretrained.
# output_dir should be something containing an adapter_config.json and adapter_model.bin:
model = PeftModel.from_pretrained(model, output_dir)


# 8. Evaluate the model
eval_prompt = generate_eval_prompt(dataset["test"][0])
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input,
          max_new_tokens=100)[0], skip_special_tokens=True))
