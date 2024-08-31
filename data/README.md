# Downloading the dataset

To download the latest version of the `FixMe` dataset or to use `repairllama` dataset, execute the following command in the root directory of the repository:

```
python3 -m source.run
```

This command will download the most recent FixMe dataset to `data/FixMe-v1.db`, as specified in the config.yaml file.

The `FixMe` database file has the information of different granularity levels open-source project repositories which can be utilized for different software security applications, i.e., automated patch prediction, automated program repair, commit classification, vulnerability prediction, etc.

Additionally, the script will automatically download and utilize the `repairllama` dataset from Huggingface using the `datasets` library. You can find more information about this dataset [here](https://huggingface.co/datasets/ASSERT-KTH/repairllama-datasets).
