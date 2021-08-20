from tqdm import tqdm
import os
import numpy as np
import yaml
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from seq2seq_model import MT5ForJudgementGeneration
from mt5_model.transformers_pkg.models.t5 import T5Tokenizer

from dataset import get_dataset

from torch.optim import AdamW
from torch.utils.data import DataLoader


def train(config):
    # 指定可用GPU数量
    device_ids = [0, 1, 2, 3]
    CURRENT_DIR = config["CURRENT_DIR"]
    train_set, test_set = get_dataset(config)
    model = MT5ForJudgementGeneration(config)
    if config["use_cuda"] and torch.cuda.is_available():
        # model = torch.nn.DataParallel(model)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        # model = model.cuda()
        model = model.cuda(device=device_ids[0])
        # model = torch.nn.parallel.DistributedDataParallel(model)
    logger.info("加载模型完成...")
    train_dataloader = DataLoader(dataset=train_set, batch_size=config["batch_size"], shuffle=True)
    eval_dataloader = DataLoader(dataset=eval_set, batch_size=config["batch_size"], shuffle=True)

    optimizer = AdamW(model.parameters(), config["LR"])
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_set))
    logger.info("  Num eval examples = %d", len(eval_set))
    # logger.info("  Num test examples = %d", len(test_dataloader)*config["batch_size"])
    logger.info("  Num Epochs = %d", config["EPOCH"])
    logger.info("  Learning rate = %d", config["LR"])

    model.train()

    for epoch in range(config["EPOCH"]):
        for index, batch in enumerate(train_dataloader):
            # print(batch)
            # break
            optimizer.zero_grad()
            input_ids, attention_mask, token_type_ids = \
                batch[0].squeeze(), batch[1].squeeze(), batch[2].squeeze()
            label = batch[3]
            if config["use_cuda"] and torch.cuda.is_available():
                input_ids, attention_mask, token_type_ids = \
                    input_ids.cuda(device=device_ids[0]), attention_mask.cuda(
                        device=device_ids[0]), token_type_ids.cuda(device=device_ids[0])
                label = label.cuda(device=device_ids[0])
            model_output = model(input_ids, attention_mask, token_type_ids)
            train_loss = cross_entropy(model_output, label)
            train_loss.backward()
            optimizer.step()
