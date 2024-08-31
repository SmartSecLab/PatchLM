"""
Copyright (C) 2023 Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the Apachee-2.0 license- http://www.apache.org/licenses/

Project: PatchT5 - Code Language Models on Generating Vulnerability Security Fixes utilizing Commit Hunks
@Programmer: Guru Bhandari
"""

import logging
import time
import subprocess
import yaml
from pathlib import Path


# Configure logger only once
if not hasattr(logging, "logger_configured"):
    logging.logger_configured = True

    # Create file handler which log even debug messages
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y-%m-%d--%H-%M-%S")
    log_filename = log_dir / f"run-{run_id}.log"

    # Create a logger
    logger = logging.getLogger("logs")
    logger.setLevel(logging.DEBUG)

    # Create file handler which log even debug messages
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Make the logger accessible globally
    logging.root = logger


def get_logger():
    """ Return the logger """
    return logging.root


def load_config():
    """ Load the configuration from the YAML file """
    config_file = "config.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    config["run_id"] = run_id
    return config


def run_os_command(command):
    try:
        result = subprocess.run(command, shell=True,
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("Output:", result.stdout)
        else:
            print("Error:", result.stderr)
    except Exception as e:
        print("An error occurred:", e)
