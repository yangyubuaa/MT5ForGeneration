from tqdm import tqdm
import os
import numpy as np
import yaml
import torch
import sys
print(sys.path)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from seq2seq_model import MT5ForJudgementGeneration
from mt5_model.transformers_pkg.models.t5.tokenization_t5 import T5Tokenizer

from dataset import get_dataset

from torch.optim import SGD
from torch.utils.data import DataLoader


def train(config):
    tokenizer = T5Tokenizer.from_pretrained("mt5_model/mt5-base")
    # 指定可用GPU数量
    device_ids = [0, 1]
    model = MT5ForJudgementGeneration(config)
    if config["use_cuda"] and torch.cuda.is_available():
        # model = torch.nn.DataParallel(model)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        # model = model.cuda()
        model = model.cuda(device=device_ids[0])
        # model = torch.nn.parallel.DistributedDataParallel(model)
    logger.info("加载模型完成...")
    logger.info("加载数据...")
    train_set, test_set = get_dataset(config)
    
    
    train_dataloader = DataLoader(dataset=train_set, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=config["batch_size"], shuffle=True)

    optimizer = SGD(model.parameters(), config["LR"])
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_set))
    logger.info("  Num test examples = %d", len(test_set))
    # logger.info("  Num test examples = %d", len(test_dataloader)*config["batch_size"])
    logger.info("  Num Epochs = %d", config["EPOCH"])
    logger.info("  Learning rate = %d", config["LR"])

    model.train()

    for epoch in range(config["EPOCH"]):
        for index, batch in enumerate(train_dataloader):
            # print(batch)
            # break
            inputs = batch[0][0]
            labels = batch[1][0]
            optimizer.zero_grad()
            if config["use_cuda"] and torch.cuda.is_available():
                inputs, labels = \
                    inputs.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])

            model_output = model(inputs, labels)
            train_loss = model_output.loss
            logger.info(train_loss)
            logits = model_output.logits
            print(logits.shape)
            output = torch.argmax(logits.cpu(), dim=2).numpy().tolist()[0]
            logger.info(tokenizer.decode(output))
            train_loss.backward()
            optimizer.step()

if __name__=="__main__":
    with open(r'config.yaml', 'r', encoding='utf-8') as f:
        result = f.read()
        config = yaml.load(result)
    train(config)