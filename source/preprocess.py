import sqlite3
import pandas as pd
from datasets import Dataset, DatasetDict


# custom functions
import source.utility as util


# Setup logger
log = util.get_logger()
config = util.load_config()


db_file = config["preprocess"]["db_file"]
# List of programming languages to include in the dataset
# CodeT5 supported: Python, Java, JavaScript, PHP, Ruby, Go, C, and C#
prog_list = config["preprocess"]["prog_lang"]
log.info("Programming languages: %s", prog_list)


def load_df_from_sqlite():
    """Load the dataset from the SQLite database"""
    conn = sqlite3.connect(db_file)

    if config["debug_mode"]:
        log.info("Debug mode is ON")
        df = pd.read_sql_query(
            "SELECT * FROM hunk_collection LIMIT 500;", conn)
    else:
        df = pd.read_sql_query("SELECT * FROM hunk_collection;", conn)

    df = df[df.programming_language.isin(prog_list)].reset_index(drop=True)
    df = df[["code_before", "code_after"]]
    log.info(f"Dataset shape: {df.shape}")
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
            "dialogue": train_df["code_before"],
            "summary": train_df["code_after"],
            "topic": [""] * len(train_df),
        }
    )
    validation_dataset = Dataset.from_dict(
        {
            "id": list(validation_df.index),
            "dialogue": validation_df["code_before"],
            "summary": validation_df["code_after"],
            "topic": [""] * len(validation_df),
        }
    )
    test_dataset = Dataset.from_dict(
        {
            "id": list(test_df.index),
            "dialogue": test_df["code_before"],
            "summary": test_df["code_after"],
            "topic": [""] * len(test_df),
        }
    )

    # Create DatasetDict with the desired format
    dataset = DatasetDict(
        {"train": train_dataset,
         "validation": validation_dataset,
         "test": test_dataset,
         }
    )
    log.info(f'Train shape: {dataset["train"].shape}')
    log.info(f'Validation shape: {dataset["validation"].shape}')
    log.info(f'Test shape: {dataset["test"].shape}')
    return dataset
