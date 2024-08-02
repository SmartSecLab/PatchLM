# Fine-tune the model on the dataset


import torch
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.peft_model import get_peft_model_state_dict
from transformers import (Trainer, TrainingArguments,
                          default_data_collator)

# custom functions
import source.utility as util

dash_line = "=" * 50

# Setup logger
log = util.get_logger()
config = util.load_config()


def tokenize_function(example, tokenizer):
    start_prompt = "Generate a fix for the following vulnerable code:\n"
    end_prompt = "\nfix:\n"
    prompt = [
        start_prompt + vulnerable + end_prompt for vulnerable in example["vulnerable"]
    ]
    example["input_ids"] = tokenizer(
        prompt, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids
    example["labels"] = tokenizer(
        example["fix"], padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids

    return example


def fine_tune_codet5_model(dataset, model, tokenizer, output_dir):
    # The dataset actually contains 3 diff splits: train, validation, test.
    # The tokenize_function code is handling all data across all splits in batches.
    tokenized_datasets = dataset.map(
        lambda example: tokenize_function(example, tokenizer), batched=True
    )
    tokenized_datasets = tokenized_datasets.remove_columns(
        [
            "id",
            "topic",
            "vulnerable",
            "fix",
        ]
    )

    # # Filter the dataset to keep only a few examples for training
    # tokenized_datasets = tokenized_datasets.filter(
    #     lambda example, index: index % 100 == 0, with_indices=True)

    log.info(f"Shapes of the datasets:")
    log.info(f"Training: {tokenized_datasets['train'].shape}")
    log.info(f"Validation: {tokenized_datasets['validation'].shape}")
    log.info(f"Test: {tokenized_datasets['test'].shape}")
    log.info(tokenized_datasets)

    num_train_epochs = config['fine_tuning']['num_train_epochs']
    batch_size = config['fine_tuning']['batch_size']
    per_device_train_batch_size = config['fine_tuning']['per_device_train_bsize']
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    if config["debug_mode"] is False:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=float(config["fine_tuning"]["learning_rate"]),
            num_train_epochs=config["fine_tuning"]["num_train_epochs"],
            weight_decay=config["fine_tuning"]["weight_decay"],
            logging_steps=config["fine_tuning"]["logging_steps"],
            max_steps=config["fine_tuning"]["max_steps"],
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=float(config["fine_tuning"]["learning_rate"]),
            gradient_accumulation_steps=gradient_accumulation_steps,
            # optim="paged_adamw_32bit",
            # save_steps=1,
            # evaluation_strategy="steps",
            # eval_steps=1,
            # fp16=True,
            # bf16=False,
            num_train_epochs=1,
            weight_decay=config["fine_tuning"]["weight_decay"],
            logging_steps=1,
            max_steps=1,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    trainer.train()

    # Save the trained model
    trainer.save_model(output_dir)
    log.info(dash_line)
    log.info("Fine-Tuning Completed!")
    log.info(f"Model saved to: {output_dir}")
    log.info(dash_line)

    return trainer


def create_peft_config(model):
    """ Create a PEFT configuration"""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    # prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


def fine_tune_codellama_model(config, model, tokenizer, tokenized_train_dataset, tokenized_val_dataset):
    """ Fine-tune the codellama model"""
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    output_dir = config['fine_tuning']['output_dir']
    # 6. Training arguments
    num_train_epochs = config['fine_tuning']['num_train_epochs']
    batch_size = config['fine_tuning']['batch_size']
    per_device_train_batch_size = config['fine_tuning']['per_device_train_bsize']
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=100,
        learning_rate=float(config['fine_tuning']['learning_rate']),
        eval_strategy="steps",
        eval_steps=100,
        fp16=True,
        bf16=False,
        group_by_length=True,
        logging_strategy="steps",
        save_strategy="no",
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator=default_data_collator,
    )

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        model, type(model)
    )

    # Train and save the model
    trainer.train()

    trainer.model.save_pretrained(output_dir)
    trainer.save_model(output_dir)
    log.info(f"Model saved to: {output_dir}")
    log.info("Fine-Tuning Completed!")
    log.info("=" * 50)
    return trainer, model, tokenizer
