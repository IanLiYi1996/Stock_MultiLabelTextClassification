#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys, json
import argparse
import numpy as np
import pandas as pd
import logging, logging.config
from ordered_set import OrderedSet
from torchsummaryX import summary
from BERT_DSSM_model import BertDSSM
from dataloader import TrainDataSet, TrainDataSet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


def get_data_loader(dataset_class, batch_size, shuffle=True, num_workers=1):
    return  DataLoader(
            dataset_class,
            batch_size      = batch_size,
            shuffle         = shuffle,
            num_workers     = max(0, num_workers),
            collate_fn      = dataset_class.collate_fn
        )

def train_model(args, dataset, model):
    for epoch in range(args.epoches):
        for step, batch_data in enumerate(iter(get_data_loader(dataset, args.batch_size))):
            query_pt, pos_pt, neg_pt, label = batch_data
            out_q = model.forward(query_pt)
            out_pos = model.forward(pos_pt)
            out_neg = model.forward(neg_pt)
            cos_qp = torch.cosine_similarity(out_q, out_pos, dim=1)
            cos_qn = torch.cosine_similarity(out_q, out_neg, dim=1)
            margin = torch.full((args.batch_size, 1), args.margin, dtype=torch.float64)
            zeros = torch.zeros((args.batch_size, 1))
            losses = cos_qn - cos_qp + margin
            # print('cos_qn:{}'.format(cos_qn))
            # print('cos_qp:{}'.format(cos_qp))
            # print('loss:{}'.format(losses))
            losses = torch.stack((losses[0].reshape(args.batch_size,1), zeros), dim=1)
            losses = torch.max(losses, dim=1).values
            # cos_uni = torch.stack((cos_qp, cos_qn), dim=1)
            # print('cos:{}'.format(cos_uni))
            # softmax_qp = F.softmax(cos_uni, dim=1)
            # print('softmax:{}'.format(softmax_qp))
            # losses = -torch.log(torch.prod(softmax_qp, dim=1))
            loss = torch.mean(losses)
            loss = loss.requires_grad_()
            print('Epoch:{}       train_loss:{}       accuracy:{}       eval_accuracy{}'.format(epoch,loss,0,0,0))
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

def accuracy():
    raise NotImplementedError()

def save_model(model, optimizer, path):
    state = {
        'state_dict'	: model.state_dict(),
        'optimizer'	: optimizer.state_dict()
    }
    torch.save(state, path)

def load(path):
    model = BertDSSM(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    return model, optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
    parser.add_argument('-pretrained_model',		default='./bert_model',					help='input the bert models file')
    parser.add_argument('-nsize',		default=768,					help='input size')
    parser.add_argument('-outfeatures',		default=128,					help='output size')
    parser.add_argument('-stock_info',		default='data/stock_info.csv',					help='data use for stock dict')
    parser.add_argument('-hiddensize',		default=256,					help='hidden units')
    parser.add_argument('-epoches',		default=10,					help='epoch num')
    parser.add_argument('-batch_size',		default=2,					help='hidden units')
    parser.add_argument('-trainfile',		default='./data/DSSM/train.csv',					help='input the train file')
    parser.add_argument('-symbol_file',		default='./data/stock_info.csv',					help='input the symbol file')
    parser.add_argument('-testfile',		default='./data/DSSM/train.csv',					help='input the test file')
    parser.add_argument('-max_seq_len',		default=128,					help='input the test file')
    parser.add_argument('-margin',		default=1.0,					help='margin value')
    parser.add_argument('-logdir',         default='./log/',               help='Log directory')
    parser.add_argument('-config',      default='./config/',            help='Config directory')
    args = parser.parse_args()
    
    logger = get_logger(args.name, args.logdir, args.config)
    model = BertDSSM(args)
    for name, parameters in model.named_parameters():
        if parameters.requires_grad:
            print(name,':',parameters.size())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_data = TrainDataSet(args)
    summary(model, torch.ones(1,768))
    # train_model(args, train_data, model)
