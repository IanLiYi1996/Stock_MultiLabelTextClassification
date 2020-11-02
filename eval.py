#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys, json
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics

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

def accuracy(y, pred):
    pred = torch.argmax(pred, dim=1)
    acc = metrics.accuracy_score(y, pred)
    return acc

def calauc(y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def save_model(model, optimizer, path):
    state = {
        'state_dict'	: model.state_dict(),
        'optimizer'	: optimizer.state_dict()
    }
    torch.save(state, path)

def load(args,path):
    model = BertDSSM(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    return model, optimizer

def eval_data():
    raise NotImplementedError()

def eval_dataset(args, model):
    raise NotImplementedError()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
    parser.add_argument('-pretrained_model',		default='./bert_model',					help='input the bert models file')
    parser.add_argument('-nsize',		default=768,					help='input size')
    parser.add_argument('-outfeatures',		default=128,					help='output size')
    parser.add_argument('-stock_info',		default='data/stock_info.csv',					help='data use for stock dict')
    parser.add_argument('-hiddensize',		default=256,					help='hidden units')
    parser.add_argument('-epoches',		default=10,					help='epoch num')
    parser.add_argument('-batch_size',		default=1,					help='hidden units')
    parser.add_argument('-trainfile',		default='./data/DSSM/train.csv',					help='input the train file')
    parser.add_argument('-symbol_file',		default='./data/stock_info.csv',					help='input the symbol file')
    parser.add_argument('-testfile',		default='./data/DSSM/train.csv',					help='input the test file')
    parser.add_argument('-max_seq_len',		default=128,					help='input the test file')
    parser.add_argument('-margin',		default=1.0,					help='margin value')
    parser.add_argument('-logdir',         default='./log/',               help='Log directory')
    parser.add_argument('-config',      default='./config/',            help='Config directory')
    parser.add_argument('-savepath',      default='./outputs/',            help='Config directory')
    args = parser.parse_args()
    
    model_path = './outputs/epoch0.pt'
    model, _ =load(args,model_path)
