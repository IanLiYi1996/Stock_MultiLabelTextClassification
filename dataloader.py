#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


class TrainDataSet(Dataset):
    """
	Training Dataset class.

	Parameters
	----------

	Returns
	-------
	A training Dataset class instance used by DataLoader
	"""
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args.trainfile, sep="\t", dtype='str')
        self.query_doc = self.get_symbol_text(self.args.symbol_file)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)
        self.encoder = AutoModel.from_pretrained(self.args.pretrained_model)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        query, pos, neg, label = sample['symbol'], sample['pos'], sample['neg'], sample['label']
        # print(query, self.query_doc[query], pos, neg)
        negs = neg.split('0x10')
        labels = [int(i) for i in label.split(' ')]
        query_pt = self.tokenize(self.query_doc[query][:self.args.max_seq_len])
        pos_pt = self.tokenize(pos[:self.args.max_seq_len])
        query_pt = self.get_embedding(query_pt)
        pos_pt = self.get_embedding(pos_pt)
        negs_pt = []
        for i in range(0, len(negs)):
            neg_pt = self.tokenize(negs[i][:self.args.max_seq_len]) 
            neg_pt = self.get_embedding(neg_pt)
            negs_pt.append(neg_pt)
        return query_pt, pos_pt, negs_pt[0],negs_pt[1], negs_pt[2],negs_pt[3], torch.tensor(labels[:5])

    def __len__(self):
        return len(self.data)

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        token_pt = torch.tensor([token_ids])
        return token_pt

    @torch.no_grad()
    def get_embedding(self, tensors):
        _, pooled = self.encoder(tensors)
        return pooled[0]

    def get_symbol_text(self, in_file):
        symbol_doc = OrderedDict()
        fin = pd.read_csv(in_file)
        for idx, row in fin.iterrows():
            symbol = str(row['symbol_id'])
            sent = str(row['main_operation_business'])+str(row['operating_scope'])+str(row['org_cn_introduction'])
            symbol_doc[symbol] =sent
        return symbol_doc

    @staticmethod
    def collate_fn(data):
        query_pt = torch.stack([_[0] 	for _ in data], dim=0)
        pos_pt = torch.stack([_[1] for _ in data], dim=0)
        neg_pt_1 = torch.stack([_[2] 	for _ in data], dim=0)
        neg_pt_2 = torch.stack([_[3] 	for _ in data], dim=0)
        neg_pt_3 = torch.stack([_[4] 	for _ in data], dim=0)
        neg_pt_4 = torch.stack([_[5] 	for _ in data], dim=0)
        # neg_pt_5 = torch.stack([_[6] 	for _ in data], dim=0)
        label = torch.stack([_[6] 	for _ in data], dim=0)
        return query_pt, pos_pt, neg_pt_1, neg_pt_2, neg_pt_3, neg_pt_4, label

class TestDataSet(Dataset):
    """
	Testing Dataset class.

	Parameters
	----------
	
	Returns
	-------
	A testing Dataset class instance used by DataLoader
	"""
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args.testfile, sep="\t", dtype='str')
        self.query_doc = self.get_symbol_text(self.args.symbol_file)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)
        self.encoder = AutoModel.from_pretrained(args.pretrained_model)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        query, pos, neg, label = sample['symbol'], sample['pos'], sample['neg'], sample['label']
        # print(query, self.query_doc[query], pos, neg)
        negs = neg.split('0x10')
        labels = label.split(' ')
        query_pt = self.tokenize(self.query_doc[query][:self.args.max_seq_len])
        pos_pt = self.tokenize(pos[:self.args.max_seq_len])
        query_pt = self.get_embedding(query_pt)
        pos_pt = self.get_embedding(pos_pt)
        negs_pt = []
        for i in range(0, len(negs)):
            neg_pt = self.tokenize(negs[i][:self.args.max_seq_len]) 
            neg_pt = self.get_embedding(neg_pt)
            negs_pt.append(neg_pt)
        return query_pt, pos_pt, negs_pt[0],negs_pt[1], negs_pt[2],negs_pt[3],negs_pt[4], torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        token_pt = torch.tensor([token_ids])
        return token_pt
    
    @torch.no_grad()
    def get_embedding(self, tensors):
        _, pooled = self.encoder(tensors)
        return pooled[0]

    def get_symbol_text(self, in_file):
        symbol_doc = OrderedDict()
        fin = pd.read_csv(in_file)
        for idx, row in fin.iterrows():
            symbol = str(row['symbol_id'])
            sent = str(row['main_operation_business'])+str(row['operating_scope'])+str(row['org_cn_introduction'])
            symbol_doc[symbol] =sent
        return symbol_doc

    @staticmethod
    def collate_fn(data):
        query_pt = torch.stack([_[0] 	for _ in data], dim=0)
        pos_pt = torch.stack([_[1] for _ in data], dim=0)
        neg_pt_1 = torch.stack([_[2] 	for _ in data], dim=0)
        neg_pt_2 = torch.stack([_[3] 	for _ in data], dim=0)
        neg_pt_3 = torch.stack([_[4] 	for _ in data], dim=0)
        neg_pt_4 = torch.stack([_[5] 	for _ in data], dim=0)
        neg_pt_5 = torch.stack([_[6] 	for _ in data], dim=0)
        label = torch.stack([_[7] 	for _ in data], dim=0)
        return query_pt, pos_pt, neg_pt_1, neg_pt_2, neg_pt_3, neg_pt_4, neg_pt_5, label


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pretrained_model',		default='./bert_model',					help='input the bert models file')
    parser.add_argument('-nsize',		default=768,					help='input size')
    parser.add_argument('-outfeatures',		default=200,					help='output size')
    parser.add_argument('-stock_emb_file',		default='./output/stock.npy',					help='output size')
    parser.add_argument('-trainfile',		default='./data/DSSM/train.csv',					help='input the train file')
    parser.add_argument('-symbol_file',		default='./data/stock_info.csv',					help='input the symbol file')
    parser.add_argument('-testfile',		default='./data/DSSM/train.csv',					help='input the test file')
    parser.add_argument('-max_seq_len',		default=128,					help='input the test file')
    args = parser.parse_args()
    
    train_data = TrainDataSet(args)
    for i in range(10):
        print(train_data[i])
    print(len(train_data))
    # dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)