"""
Copyright (C) 2023 Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the Apachee-2.0 license- http://www.apache.org/licenses/

Project: PatchT5 - Code Language Models on Generating Vulnerability Security Fixes utilizing Commit Hunks
@Programmer: Guru Bhandari
"""

import os
import requests
import sqlite3
import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import load_dataset

# custom functions
import source.utility as util


# Setup logger
log = util.get_logger()
config = util.load_config()
max_hunks_per_url = config["preprocess"]["max_hunks_per_url"]

db_file = config["preprocess"]["db_file"]
# List of programming languages to include in the dataset
# CodeT5 supported: Python, Java, JavaScript, PHP, Ruby, Go, C, and C#
prog_list = config["preprocess"]["prog_lang"]
# log.info("Programming Languages: %s", prog_list)


def filter_patches(df_patch, max_hunks_per_url=2):
    """Filter URLs with counts less than max_hunks_per_url"""
    # Calculate value counts of 'url' column
    url_counts = df_patch["url"].value_counts()
    urls_less_than_two = url_counts[url_counts <=
                                    max_hunks_per_url].index.tolist()
    df = df_patch[df_patch.url.isin(urls_less_than_two)]

    # print(f'Shape of filtered patch data: {df.shape}')
    return df


def filter_hunks(df_hunk, df_patch):
    """Filter hunks that are not in the filtered patches"""
    df_hunk = df_hunk[df_hunk.file.isin(df_patch.file)]
    # print(f'Shape of filtered hunk data: {df_hunk.shape}')
    return df_hunk


def download_file_if_not_exists(file_path, file_url):
    """
    Check if the file exists at file_path, if not, download it from file_url.
    """
    if not os.path.exists(file_path):
        log.info(f'Downloading the file from: {file_url}\n')
        # If not, download the file
        response = requests.get(file_url)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Write the content to the file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        log.info(f'File downloaded and saved to {file_path}')
    else:
        log.info(f'File already exists at {file_path}')


def load_df_from_sqlite():
    """Load the dataset from the SQLite database"""
    file_url = 'https://zenodo.org/records/10955342/files/FixMe-v1.db?download=1'
    download_file_if_not_exists(db_file, file_url)

    conn = sqlite3.connect(db_file)
    df_hunk = pd.read_sql_query("SELECT * FROM hunk_collection;", conn)
    df_patch = pd.read_sql_query("SELECT * FROM patch_collection;", conn)

    if df_hunk.empty or df_patch.empty:
        log.error("No data found in the database")
        exit(1)

    if max_hunks_per_url is not None:
        df_patch = filter_patches(df_patch, max_hunks_per_url)
        df_hunk = filter_hunks(df_hunk, df_patch)
    else:
        log.info("No filter applied")

    if config["debug_mode"]:
        log.info("Debug mode is ON")
        df_hunk = df_hunk.sample(500, random_state=41)

    df = df_hunk[df_hunk.programming_language.isin(
        prog_list)].reset_index(drop=True)

    # put topic from patch_collection to hunk_collection comparing file_id
    df = df.merge(df_patch[["file_id", "message"]], on="file_id", how="inner")

    # Rename the columns
    df = df.rename(columns={"message": "topic"})
    df = df[["code_before", "code_after", "topic", "programming_language"]]

    log.info(f"Dataset shape: {df.shape}")
    log.info(f"Columns in hunk_collection: \n{df.columns}\n")
    return df


def add_question(example):
    """ Add a new feature- question to the dataset """
    if "question" not in example:
        example[
            "question"
        ] = "What is the fix version of the code for the following vulnerability?"
    return example


def load_dataset_from_fixme():
    """Load the dataset and split it into train, val, and test sets"""
    df = load_df_from_sqlite()
    # Verify expected columns
    required_columns = ["code_before", "code_after",
                        "topic", "programming_language"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column in the DataFrame: {col}")

    # Convert the DataFrame into a Dataset
    dataset = Dataset.from_dict(
        {
            "id": list(df.index),
            "vulnerable": df["code_before"].tolist(),
            "fix": df["code_after"].tolist(),
            "topic": df["topic"].tolist(),
            "programming_language": df["programming_language"].tolist(),
        }
    )

    # Split the dataset into train, validation, and test
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    test_val_split = train_test_split["test"].train_test_split(
        test_size=0.5, seed=42)

    dataset = DatasetDict(
        {
            "train": train_test_split["train"],
            "validation": test_val_split["train"],
            "test": test_val_split["test"],
        }
    )
    dataset = dataset.map(add_question)
    log.info(f'Train shape: {dataset["train"].shape}')
    log.info(f'Validation shape: {dataset["validation"].shape}')
    log.info(f'Test shape: {dataset["test"].shape}')
    print('Dataset loaded successfully!')
    log.info("=" * 50)
    return dataset

### ================================== ###
# load repairllama dataset


def prepare_examples(dataset):
    """ Similarize the dataset by adding a question to the dataset
    and renaming the columns"""
    dataset = dataset.map(add_question)
    # add programming_language column
    dataset = dataset.map(
        lambda x: {"programming_language": "Java"})
    # print(dataset)
    # rename the columns
    dataset = dataset.rename_column("input", "vulnerable")
    dataset = dataset.rename_column("output", "fix")
    # Add a validation split (e.g., 10% of the training data)
    if "validation" not in dataset.keys():
        train_test_split = dataset["train"].train_test_split(
            test_size=0.2, seed=42)
        test_val_split = train_test_split["test"].train_test_split(
            test_size=0.5, seed=42)
        dataset["train"] = train_test_split["train"]
        # 10% of the training data
        dataset["validation"] = test_val_split["train"]
        dataset["test"] = test_val_split["test"]  # 10% of the training data
    return dataset


def load_repairllama_dataset():
    """ Load the repairllama dataset """
    dataset = load_dataset(
        "ASSERT-KTH/repairllama-datasets", "ir1xor1", cache_dir="data/repairllama"
    )
    # Load irXxorY
    # dataset = load_dataset("ASSERT-KTH/repairllama-dataset", "irXxorY")
    log.info("=" * 50)
    log.info("Loading the dataset...")

    # Limit the dataset to 500 samples for debugging
    debug_size = 500
    if config["debug_mode"]:
        log.info("Debug mode is ON")
        # Shuffle and select 500 samples for each split
        dataset = {
            split: dataset[split].shuffle(seed=42).select(range(debug_size))
            for split in dataset.keys()
        }
        dataset = DatasetDict(dataset)
    else:  # Shuffle the dataset
        log.info("Debug mode is OFF")
        # dataset = {split: dataset[split].shuffle(seed=42).select(
        #     range(999)) for split in dataset.keys()}
        # dataset = DatasetDict(dataset)

    dataset = prepare_examples(dataset)
    log.info(dataset)
    log.info(f"Dataset shape: {dataset.shape}")
    # example = dataset["train"][0]
    # log.info(f"Example: \n{example}")
    # # split 'test' set into test and validation set
    test_val_splits = dataset['test'].train_test_split(test_size=0.5, seed=42)
    dataset = DatasetDict({
        'train': dataset['train'], 
        'validation': test_val_splits['train'],
        'test': test_val_splits['test']
    })
    log.info("=" * 50)
    return dataset
