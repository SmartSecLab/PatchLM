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
log.info("Programming Languages: %s", prog_list)


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


def load_dataset_from_df():
    """Load the dataset and split it into train, val, and test sets"""
    df = load_df_from_sqlite()
    total_rows = len(df)
    train_size = int(total_rows * 0.8)
    val_size = int(total_rows * 0.1)

    train_df = df.iloc[:train_size]
    validation_df = df.iloc[train_size: train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    # Create Dataset objects
    train_dataset = Dataset.from_dict(
        {
            "id": list(train_df.index),
            "vulnerable": train_df["code_before"],
            "fix": train_df["code_after"],
            "topic": train_df["topic"],
            "programming_language": train_df["programming_language"],
        }
    )
    validation_dataset = Dataset.from_dict(
        {
            "id": list(validation_df.index),
            "vulnerable": validation_df["code_before"],
            "fix": validation_df["code_after"],
            "topic": validation_df["topic"],
            "programming_language": validation_df["programming_language"],
        }
    )
    test_dataset = Dataset.from_dict(
        {
            "id": list(test_df.index),
            "vulnerable": test_df["code_before"],
            "fix": test_df["code_after"],
            "topic": test_df["topic"],
            "programming_language": test_df["programming_language"],
        }
    )

    # Create DatasetDict with the desired format
    dataset = DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset,
        }
    )
    log.info(f'Train shape: {dataset["train"].shape}')
    log.info(f'Validation shape: {dataset["validation"].shape}')
    log.info(f'Test shape: {dataset["test"].shape}')
    print('Dataset loaded successfully!')
    log.info("=" * 50)
    return dataset

### ================================== ###
# load repairllama dataset


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
    dataset = dataset.rename_column("input", "vulnerable")
    dataset = dataset.rename_column("output", "fix")
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
    dataset = prepare_examples(dataset)
    log.info(dataset)
    log.info("=" * 50)
    return dataset
