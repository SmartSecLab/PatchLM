# PatchLM: Generating Vulnerability Security Fixes with Code Language Models

Welcome to the repository for PatchLM, a novel code language model fine-tuned specifically to fix security vulnerabilities in code blocks derived from commit hunks associated with Common Vulnerabilities and Exposures (CVE) records.
By leveraging our model to understand secure coding practices and generate accurate patches, PatchLM aims to address the diverse nature of security flaws across multiple programming languages.
Our experimental evaluation demonstrates that PatchLM significantly outperforms the baseline `CodeT5` and `CodeLlama` models in generating effective security patches, as reflected in the performance metrics.
This research highlights the potential of using code language models (CLM) to enhance automated vulnerability fixing, offering a promising path for future advancements in the field.

# Configuration Parameters

We provide information on the configuration parameters used for inference, including model settings (see `config.yaml`) and prompt structures (see `source/prompt.py`).

# Running the Code

Running the command will complete the CLM pipeline; download the specified dataset, preprocess the data, fine-tune the models and evaluate the fine-tuned intruct model with the base model.

```
python3 -m source.run
```

# Dependencies

Before generating patches, ensure that you have the necessary dependencies installed. The work is programmed in Python 3.8.0 and it requires several Python libraries as specified in `requirements.txt`:

# Dataset

The used `FixMe` dataset is available at [Zenodo](https://zenodo.org/records/10955342). Please refer [GitHub repository](https://github.com/SmartSecLab/FixMe) to construct updated version of the dataset. Additionally, we have utilized `repairllama` dataset from Huggingface using the `datasets` library. You can find more information about this dataset [here](https://huggingface.co/datasets/ASSERT-KTH/repairllama-datasets).

# Citation

If you use this repository or PatchLM in your research or project, please cite the following paper:

```
@article{Bhandari2025,
  title={Generating vulnerability security fixes with Code Language Models},
  author={Bhandari, Guru and Gavric, Nikola and Shalaginov, Andrii},
  journal={Information and Software Technology},
  volume={185},
  year={2025},
  doi={10.1016/j.infsof.2025.107786}
}

```

# Acknowledgement

The data extraction process received substantial support from the `Kristiania-HPC` infrastructure hosted at Kristiania University College. Additionally, the machine learning experiments in this research study were made possible by the `eX3` HPC infrastructure hosted in Simula Research Laboratory.
