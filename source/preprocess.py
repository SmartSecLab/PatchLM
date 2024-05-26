import sqlite3
import pandas as pd
from datasets import Dataset, DatasetDict


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
    url_counts = df_patch['url'].value_counts()
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


def load_df_from_sqlite():
    """Load the dataset from the SQLite database"""
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
            "vulnerable": train_df["code_before"],
            "fix": train_df["code_after"],
            "context": [""] * len(train_df),
        }
    )
    validation_dataset = Dataset.from_dict(
        {
            "id": list(validation_df.index),
            "vulnerable": validation_df["code_before"],
            "fix": validation_df["code_after"],
            "context": [""] * len(validation_df),
        }
    )
    test_dataset = Dataset.from_dict(
        {
            "id": list(test_df.index),
            "vulnerable": test_df["code_before"],
            "fix": test_df["code_after"],
            "context": [""] * len(test_df),
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
