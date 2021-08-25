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

from torch.optim import Adam
from torch.utils.data import DataLoader

from torch.nn.functional import cross_entropy


def train(config):
    print(config["use_cuda"] and torch.cuda.is_available())
    tokenizer = T5Tokenizer.from_pretrained("mt5_model/mt5-small-simplify")
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

    optimizer = Adam(model.parameters(), config["LR"])
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    loss_f = cross_entropy

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_set))
    logger.info("  Num test examples = %d", len(test_set))
    # logger.info("  Num test examples = %d", len(test_dataloader)*config["batch_size"])
    logger.info("  Num Epochs = %d", config["EPOCH"])
    logger.info("  Learning rate = {}".format(str(config["LR"])))

    model.train()

    for epoch in range(config["EPOCH"]):
        for index, batch in enumerate(train_dataloader):
            # print(batch)
            # break
            inputs = batch[0].squeeze()
            # print(inputs.shape)
            # print(batch[0])
            # print(inputs.shape)
            labels = batch[1].squeeze()
            # print(labels.shape)
            optimizer.zero_grad()
            if config["use_cuda"] and torch.cuda.is_available():
                inputs, labels = \
                    inputs.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])

            model_output = model(inputs, labels)
            logits = model_output.logits
            logits = logits.permute(0, 2, 1)
            # print(logits.shape)
            # print(labels.shape)
            train_loss = cross_entropy(logits, labels)
            print(train_loss)
            # print(torch.argmax(logits.cpu(), dim=1).shape)
            output = torch.argmax(logits.cpu(), dim=1).numpy().tolist()[0]
            logger.info(tokenizer.decode(output))
            train_loss.backward()
            optimizer.step()


if __name__=="__main__":
    with open(r'config.yaml', 'r', encoding='utf-8') as f:
        result = f.read()
        config = yaml.load(result)
    train(config)