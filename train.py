#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys, json
import argparse
import numpy as np
import pandas as pd
import logging, logging.config
from ordered_set import OrderedSet
from .BERT_DSSM_model import BertDSSM
from .dataloader import TrainDataSet, TrainDataSet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def get_data_loader(dataset_class, batch_size, shuffle=True, num_workers=1):
    return  DataLoader(
            dataset_class,
            batch_size      = batch_size,
            shuffle         = shuffle,
            num_workers     = max(0, num_workers),
            collate_fn      = dataset_class.collate_fn
        )

def train_model(args, dataset):
    for epoch in range(args.epoches):
        for step, batch_data in enumerate(iter(get_data_loader(dataset, args.batch_size))):
            query_pt, pos_pt, neg_pt, label = batch_data
            out_q = model.forward(query_pt)
            out_pos = model.forward(pos_pt)
            out_neg = model.forward(neg_pt)
            cos_qp = torch.cosine_similarity(out_q, out_pos, dim=1)
            cos_qn = torch.cosine_similarity(out_q, out_neg, dim=1)
            cos_uni = torch.cat((cos_qp, cos_qn), 1)
            softmax_qp = F.softmax(cos_uni, dim=1)[:, 0]
            loss = -torch.log(torch.prod(softmax_qp))
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def get_logger(name, log_dir, config_dir):
    config_dict = json.load(open( config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)
    
    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)
    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pretrained_model',		default='./bert_model',					help='input the bert models file')
    parser.add_argument('-nsize',		default=768,					help='input size')
    parser.add_argument('-outfeatures',		default=200,					help='output size')
    parser.add_argument('-stock_info',		default='data/stock_info.csv',					help='data use for stock dict')
    parser.add_argument('-hiddensize',		default=256,					help='hidden units')
    parser.add_argument('-epoches',		default=1,					help='epoch num')
    parser.add_argument('-batch_size',		default=1,					help='hidden units')
    parser.add_argument('-trainfile',		default='./data/train.csv',					help='input the train file')
    parser.add_argument('-symbol_file',		default='./data/stock_info.csv',					help='input the symbol file')
    parser.add_argument('-testfile',		default='./data/test.csv',					help='input the test file')
    args = parser.parse_args()
    
    model = BertDSSM(args)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    train_data = TrainDataSet(args)

    # train_model(args, train_data)
