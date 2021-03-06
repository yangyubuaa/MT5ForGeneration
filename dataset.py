import os
import torch
import yaml
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from tqdm import tqdm
from torch.utils.data import Dataset

from mt5_model.transformers_pkg.models.t5 import T5Tokenizer


class JudgementGenerationDataset(Dataset):
    def __init__(self, tokenized):
        super(JudgementGenerationDataset, self).__init__()
        self.tokenized = tokenized

    def __getitem__(self, item):
        return self.tokenized[item][0], self.tokenized[item][1]

    def __len__(self):
        return len(self.tokenized)


def get_dataset(config):
    cache_train_path = config["cache_train_path"]
    cache_test_path = config["cache_test_path"]

    t5model_path = config["abs_path"] + config["t5model_path"]
    tokenizer = T5Tokenizer.from_pretrained(t5model_path)

    train_set_path = config["abs_path"] + config["train_path"]
    test_set_path = config["abs_path"] + config["test_path"]

    if os.path.exists(cache_train_path):
        logger.info("train cache exists...load...")
        train_set_tokenized = torch.load(cache_train_path)
    else:
        logger.info("train cache not exists...construct...")
        train_set = read_data_file(train_set_path)
        train_set_tokenized = transfer_set2index(tokenizer, train_set)
        logger.info("train cache construce complete! save files...")
        torch.save(train_set_tokenized, cache_train_path)

    if os.path.exists(cache_test_path):
        logger.info("test cache exists...load...")
        test_set_tokenized = torch.load(cache_test_path)
    else:
        logger.info("test cache not exists...construce...")
        test_set = read_data_file(test_set_path)
        test_set_tokenized = transfer_set2index(tokenizer, test_set)
        logger.info("test cache construce complete! save files...")
        torch.save(test_set_tokenized, cache_test_path)

    return JudgementGenerationDataset(train_set_tokenized), JudgementGenerationDataset(test_set_tokenized)


def read_data_file(data_path):
    with open(data_path, "r", encoding="utf-8") as r:
        datas = [i.strip() for i in r.readlines()]
    return datas


def transfer_set2index(tokenizer, dataset):
    tokenized = list()
    for line in tqdm(dataset):
        query, value = line.split("|\t|")
        query_tokenized, value_tokenized = tokenizer(query, return_tensors="pt", max_length=512, padding="max_length", truncation=True).input_ids, tokenizer(value, return_tensors="pt", max_length=100, padding="max_length", truncation=True).input_ids
        tokenized.append([query_tokenized, value_tokenized])

    return tokenized


if __name__ == '__main__':
    with open(r'config.yaml', 'r', encoding='utf-8') as f:
        result = f.read()
        config = yaml.load(result)
    get_dataset(config)
