#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


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
        self.data = pd.read_csv(self.args.trainfile)
        self.query_doc = self.get_symbol_text(self.args.symbol_file)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        query, pos, neg, label = sample['symbol'], sample['pos'], sample['neg'], sample['label']
        query_pt = self.tokenize(self.query_doc[query])
        pos_pt = self.tokenize(pos)
        neg_pt = self.tokenize(neg)

        return query_pt, pos_pt, neg_pt, label

    def __len__(self):
        return len(self.data)

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        token_pt = torch.tensor([token_ids])
        return token_pt

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
        neg_pt = torch.stack([_[2] 	for _ in data], dim=0)
        label = torch.stack([_[3] 	for _ in data], dim=0)
        return query_pt, pos_pt, neg_pt, label

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
        self.data = pd.read_csv(self.args.testfile)
        self.query_doc = self.get_symbol_text(self.args.symbol_file)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        query, pos, neg, label = sample['symbol'], sample['pos'], sample['neg'], sample['label']
        query_pt = self.tokenize(self.query_doc[query])
        pos_pt = self.tokenize(pos)
        neg_pt = self.tokenize(neg)

        return query_pt, pos_pt, neg_pt, label

    def __len__(self):
        return len(self.data)

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        token_pt = torch.tensor([token_ids])
        return token_pt

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
        neg_pt = torch.stack([_[2] 	for _ in data], dim=0)
        label = torch.stack([_[3] 	for _ in data], dim=0)
        return query_pt, pos_pt, neg_pt, label


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pretrained_model',		default='./bert_model',					help='input the bert models file')
    parser.add_argument('-nsize',		default=768,					help='input size')
    parser.add_argument('-outfeatures',		default=200,					help='output size')
    parser.add_argument('-stock_emb_file',		default='./output/stock.npy',					help='output size')
    parser.add_argument('-trainfile',		default='./data/train.csv',					help='input the train file')
    parser.add_argument('-symbol_file',		default='./data/stock_info.csv',					help='input the symbol file')
    parser.add_argument('-testfile',		default='./data/test.csv',					help='input the test file')
    args = parser.parse_args()
    
    train_data = TrainDataSet(args)
    print(train_data[0])
    print(len(train_data))
    # dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)