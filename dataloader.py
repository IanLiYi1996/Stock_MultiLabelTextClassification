#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataSet, DataLoader
import torch.utils.tensorboard as tensorboard
from transformers import AutoTokenizer


class TrainDataSet(DataSet):
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
        sample = self.data[index]
        query, pos, neg, label = sample['symbol'], sample['pos'], sample['neg'], sample['label']
        query_pt = self.tokenize(self.query_doc[query])
        pos_pt = self.tokenize(pos)
        neg_pt = self.tokenize(neg)

        return query_pt, pos_pt, neg_pt, label

    def __len__(self):
        return len(self.data)

    def get_stock_embed(self, stock_name):
        stock_id = 0
        for idx, stock in enumerate(self.stock_set):
            if stock == stock_name:
                stock_id = idx
        return self.stock_emb[stock_id]

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        token_pt = torch.tensor([token_ids])
        return token_pt

    def get_symbol_text(self, in_file):
        symbol_doc = OrderedDict()
        fin = codecs.open(in_file, 'r', encoding='utf-8')
        for line in fin.readlines():
            line = line.strip()
            symbol_doc[line.split('\t')[0]] = line.split('\t')[1]
        return symbol_doc

class TestDataSet(DataSet):
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
        self.data = pd.read_csv(self.args.trainfile)
        self.query_doc = self.get_symbol_text(self.args.symbol_file)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)

    def __getitem__(self, index):
        sample = self.data[index]
        query, pos, neg, label = sample['symbol'], sample['pos'], sample['neg'], sample['label']
        query_pt = self.tokenize(self.query_doc[query])
        pos_pt = self.tokenize(pos)
        neg_pt = self.tokenize(neg)

        return query_pt, pos_pt, neg_pt, label

    def __len__(self):
        return len(self.data)

    def get_stock_embed(self, stock_name):
        stock_id = 0
        for idx, stock in enumerate(self.stock_set):
            if stock == stock_name:
                stock_id = idx
        return self.stock_emb[stock_id]

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        token_pt = torch.tensor([token_ids])
        return token_pt

    def get_symbol_text(self, in_file):
        symbol_doc = OrderedDict()
        fin = codecs.open(in_file, 'r', encoding='utf-8')
        for line in fin.readlines():
            line = line.strip()
            symbol_doc[line.split('\t')[0]] = line.split('\t')[1]
        return symbol_doc

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pretrained_model',		default='./bert_model',					help='input the bert models file')
    parser.add_argument('-nsize',		default=768,					help='input size')
    parser.add_argument('-outfeatures',		default=200,					help='output size')
    parser.add_argument('-stock_emb_file',		default='./output/stock.npy',					help='output size')
    parser.add_argument('-outfeatures',		default=200,					help='output size')
    args = parser.parse_args()
    # dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)