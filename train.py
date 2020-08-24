import logging
import os
import sys
import argparse

import torch
import torch.nn as nn
import pandas as pd 
import numpy as np

from toolkit.clean_text import TextProcess, DataClean
from ordered_set import OrderedSet

model = Bert_MLTC()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

class Preprocess():
    def __init__(self, params):
        """
        Constructor of the preprocess class

        Parameters
        ----------
        params:         List of hyper-parameters of the model
        
        Returns
        -------
        Creates computational graph and optimizer
        
        """
        self.p			= params 


    def load_data(self):
        """
        Reading in raw quadples and converts it into a standard format. 

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset
        
        Returns
        -------

        """
        pd.read_csv(self.p.dataset, usecols=['title','text','stock_name','label'])
        for index, row in stock.iterrows():
            title = row['title']
            text = row['text']
            stock = row['stock_name']
            label = row['label']
            if title is not 'nan':
                title = procText.process_text(cleaner.clean_text(str(title)))
            if text is not 'nan':
                text = procText.process_text(cleaner.clean_text(str(text))).split('。')
            
    def get_stockSet(self):
        stock_dict = OrderedSet()
        pd.read_csv(self.p.stock_info, usecols=['name'])
        for index, row in stock.iterrows():
            stock_dict.add(row['name'])
        return stock_set

def train(epoch):
    for i, data in enumerate(dataLoader, 0):
        x, y = data
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pretrained_model',		default='./bert_model',					help='input the bert models file')
    parser.add_argument('-nsize',		default=768,					help='input size')
    parser.add_argument('-outfeatures',		default=200,					help='output size')
    parser.add_argument('-stock_emb_file',		default='./output/stock.npy',					help='pre-embd')
    parser.add_argument('-data',		default='data/all_stock_news_label.csv',					help='dataset use for calc')
    parser.add_argument('-stock_info',		default='data/stock_info.csv',					help='data use for stock dict')
    args = parser.parse_args()
    proc = Preprocess(args)