# PatchT5: Code Language Models on Generating Vulnerability Security Fixes

Welcome to the repository for PatchT5, a novel code language model fine-tuned specifically to fix security vulnerabilities in code blocks derived from commit hunks associated with Common Vulnerabilities and Exposures (CVE) records.
By leveraging our model to understand secure coding practices and generate accurate patches, PatchT5 aims to address the diverse nature of security flaws across multiple programming languages.
Our experimental evaluation demonstrates that PatchT5 significantly outperforms the baseline CodeT5 model in generating effective security patches, as reflected in the performance metrics.
This research highlights the potential of using code language models (CLM) to enhance automated vulnerability fixing, offering a promising path for future advancements in the field.

# Structure

| File          | Description                                              |
| ------------- | -------------------------------------------------------- |
| evaluate.py   | Contains code for evaluating the performance of a model. |
| finetune.py   | Script for fine-tuning a model.                          |
| config.yaml   | YAML configuration file for generating configurations.   |
| preprocess.py | Code for preprocessing data.                             |
| prompt.py     | Script for generating prompts.                           |
| run.py        | Main script for running the model.                       |

# Configuration Parameters

We provide information on the configuration parameters used for inference, including model settings (see `config.yaml`) and prompt structures (see `source/prompt.py`).

# Dependencies

Before generating patches, ensure that you have the necessary dependencies installed. The work is programmed in Python 3.8.0 and it requires several Python libraries as specified in `requirements.txt`:

- numpy==1.21.5
- torch==1.13.1
- pandas==1.1.5
- datasets==2.19.1
- PyYAML==6.0.1
- transformers==4.40.2

# Dataset

The used dataset is available at [Zenodo](https://zenodo.org/records/5825618). Please refer [GitHub repository](https://github.com/SmartSecLab/FixMe) to construct updated version of the dataset.

# Acknowledgement

The data extraction process received substantial support from the Kristiania-HPC infrastructure hosted at Kristiania University College. Additionally, the machine learning experiments in this research study were made possible by the eX3 HPC infrastructure hosted in Simula Research Laboratory.
