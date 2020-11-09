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

torch.manual_seed(123456)


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

def load(args, path):
    model = BertDSSM(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    return model, optimizer


def loss_pairwise(margin, score_pos, score_neg):
    margin = torch.full((1, 1), margin, dtype=torch.float64)
    loss = score_neg - score_pos + margin
    zero = torch.zeros_like(loss)
    zero = zero.double()
    zero = zero.requires_grad_()
    loss = torch.max(loss, zero)
    loss = torch.mean(loss)
    loss = loss.requires_grad_()
    return loss


def train_model(args, dataset, model):
    ''' 
    pairwise
    '''
    for epoch in range(args.epoches):
        for step, batch_data in enumerate(iter(get_data_loader(dataset, args.batch_size))):
            query_pt, pos_pt, neg_pt, label = batch_data
            out_q = model.forward(query_pt)
            out_pos = model.forward(pos_pt)
            out_neg = model.forward(neg_pt)
            cos_qp = torch.cosine_similarity(out_q, out_pos, dim=1)
            cos_qp = torch.full_like(cos_qp, 0.5) + 0.5 * cos_qp
            cos_qn = torch.cosine_similarity(out_q, out_neg, dim=1)
            cos_qn = torch.full_like(cos_qn, 0.5) + 0.5 * cos_qn
            cos_uni = torch.stack((cos_qp, cos_qn), dim=1)
            pred = torch.argmax(cos_uni, dim=1)
            acc = calauc(label.tolist(), pred.tolist())
            margin = torch.full((args.batch_size, 1), args.margin, dtype=torch.float64)
            losses = cos_qn - cos_qp + margin
            zero = torch.zeros_like(losses)
            zero = zero.float()
            zero = zero.requires_grad_()
            # print('cos_qn:{}'.format(cos_qn))
            # print('cos_qp:{}'.format(cos_qp))
            # print('loss:{}'.format(losses))
            # cos_uni = torch.stack((cos_qp, cos_qn), dim=1)
            # print('cos:{}'.format(cos_uni))
            # softmax_qp = F.softmax(cos_uni, dim=1)
            # print('softmax:{}'.format(softmax_qp))
            # losses = -torch.log(torch.prod(softmax_qp, dim=1))
            losses = torch.max(losses, zero)
            loss = torch.mean(losses)
            loss = loss.requires_grad_()
            print('Epoch:{}       train_loss:{}       accuracy:{}       eval_accuracy{}'.format(epoch,loss, acc, 0))
            optimizer.zero_grad()   
            loss.backward()
            optimizer.step()
        path = args.savepath+'epoch'+str(epoch)+'.pt'
        save_model(model, optimizer, path)


def train_model_listwise(args, dataset, model,optimizer):
    ''' 
    listwise
    '''
    for epoch in range(args.epoches):
        for step, batch_data in enumerate(iter(get_data_loader(dataset, args.batch_size))):
            query_pt, pos_pt, neg_pt_1, neg_pt_2, neg_pt_3, neg_pt_4, label = batch_data
            out_q = model.forward(query_pt)
            out_pos = model.forward(pos_pt)
            out_neg_1 = model.forward(neg_pt_1)
            out_neg_2 = model.forward(neg_pt_2)
            out_neg_3 = model.forward(neg_pt_3)
            out_neg_4 = model.forward(neg_pt_4)
            cos_qp = torch.cosine_similarity(out_q, out_pos, dim=1)
            cos_qp = torch.full_like(cos_qp, 0.5) + 0.5 * cos_qp
            cos_qn1 = torch.cosine_similarity(out_q, out_neg_1, dim=1)
            cos_qn1 = torch.full_like(cos_qn1, 0.5) + 0.5 * cos_qn1
            cos_qn2 = torch.cosine_similarity(out_q, out_neg_2, dim=1)
            cos_qn2 = torch.full_like(cos_qn2, 0.5) + 0.5 * cos_qn2
            cos_qn3 = torch.cosine_similarity(out_q, out_neg_3, dim=1)
            cos_qn3 = torch.full_like(cos_qn3, 0.5) + 0.5 * cos_qn3
            cos_qn4 = torch.cosine_similarity(out_q, out_neg_4, dim=1)
            cos_qn4 = torch.full_like(cos_qn4, 0.5) + 0.5 * cos_qn4
            cos_uni = torch.stack((cos_qp, cos_qn1, cos_qn2, cos_qn3, cos_qn4), dim=1)
            print('Scores: {}'.format(cos_uni.item()))
            l1 = loss_pairwise(0.3,cos_qp, cos_qn1)
            l2 = loss_pairwise(0.4,cos_qp, cos_qn2)
            l3 = loss_pairwise(0.5,cos_qp, cos_qn3)
            l4 = loss_pairwise(0.6,cos_qp, cos_qn4)
            # dif1 = torch.sub(cos_qn1, cos_qp)
            # # print('diff {} .size{}'.format(dif1, dif1.size()))
            # dif2 = torch.sub(cos_qn2, cos_qp)
            # dif3 = torch.sub(cos_qn3, cos_qp)
            # dif4 = torch.sub(cos_qn4, cos_qp)
            # l1 = torch.log(1+torch.exp(dif1))
            # l2 = torch.log(1+torch.exp(dif2))
            # l3 = torch.log(1+torch.exp(dif3))
            # l4 = torch.log(1+torch.exp(dif4))
            l = torch.stack((l1, l2, l3, l4), dim=0)
            loss = torch.sum(l)
            loss = loss.requires_grad_()
            # print(label[0], softmax_qp[0])
            y = label.clone()
            pred = cos_uni.clone()
            auc = calauc(y[0].tolist(), pred[0].tolist())
            print('Epoch:{}   step:{}   train_loss:{}   auc:{}'.format(epoch, step,loss,auc))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        path = args.savepath+'epoch'+str(epoch)+'.pt'
        save_model(model, optimizer, path)


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
    parser.add_argument('-trainfile',		default='./data/DSSM_s2t/train.csv',					help='input the train file')
    parser.add_argument('-symbol_file',		default='./data/stock_info.csv',					help='input the symbol file')
    parser.add_argument('-testfile',		default='./data/DSSM_s2t/train.csv',					help='input the test file')
    parser.add_argument('-max_seq_len',		default=128,					help='input the test file')
    parser.add_argument('-margin',		default=1.0,					help='margin value')
    parser.add_argument('-logdir',         default='./log/',               help='Log directory')
    parser.add_argument('-config',      default='./config/',            help='Config directory')
    parser.add_argument('-savepath',      default='./outputs/',            help='Config directory')
    args = parser.parse_args()
    
    logger = get_logger(args.name, args.logdir, args.config)
    model = BertDSSM(args)
    # for name, parameters in model.named_parameters():
    #     if parameters.requires_grad:
    #         print(name,':',parameters.size())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_data = TrainDataSet(args)
    summary(model, torch.ones(1,768))
    train_model_listwise(args, train_data, model, optimizer)
    # train_model(args, train_data, model)