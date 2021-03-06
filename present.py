#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict

import logging, logging.config
from ordered_set import OrderedSet
from torchsummaryX import summary
from BERT_DSSM_model import BertDSSM
from dataloader import TrainDataSet, TrainDataSet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./bert_model')
encoder = AutoModel.from_pretrained('./bert_model')


def load_model(args,path):
    model = BertDSSM(args)
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    return model

def load_dataset(args):
    data = pd.read_csv(args.dataset, sep="\t", dtype='str')
    query_doc = get_symbol_text(args.symbol_file)
    return data, query_doc
    # for index in range(0, len(data)):
    #     sample = data.iloc[index]
    #     query, pos, neg, label = sample['symbol'], sample['pos'], sample['neg'], sample['label']
    
def tokenize(text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
    token_pt = torch.tensor([token_ids])
    _, pooled = encoder(token_pt)
    return pooled[0]

def get_scores(args, data, query_doc, model, fileout):
    fout = codecs.open(fileout, 'w', encoding='utf-8')
    fout.write('symbol\t--\tlabel\t--\tpos\t--\tneg'+'\n')
    for index in range(0, len(data)):
        sample = data.iloc[index]
        query, pos, neg = sample['symbol'], sample['pos'], sample['neg']
        label = '1-0'
        neg = neg.split('0x10')[1]
        query_pt = tokenize(query_doc[query][:args.max_seq_len])
        pos_pt = tokenize(pos[:args.max_seq_len])
        neg_pt = tokenize(neg[:args.max_seq_len])
        out_q = model.forward(query_pt)
        out_pos = model.forward(pos_pt)
        out_neg = model.forward(neg_pt)
        cos_qp = torch.cosine_similarity(out_q, out_pos, dim=0)
        cos_qp = torch.full_like(cos_qp, 0.5) + 0.5 * cos_qp
        cos_qn = torch.cosine_similarity(out_q, out_neg, dim=0)
        cos_qn = torch.full_like(cos_qn, 0.5) + 0.5 * cos_qn
        # print(cos_qp, cos_qn)
        
        print(cos_qp.items(), cos_qn.item())
        out_sample = query+'\t--\t'+str(cos_qp)+' '+str(cos_qn) +'\t--\t'+pos+'\t--\t'+neg+'\n'
        fout.write(out_sample)
        
def predict_one(args, query, text, query_doc, model):
    query_pt = tokenize(query_doc[query][:args.max_seq_len])
    text_pt = tokenize(text[:args.max_seq_len])
    out_q = model.forward(query_pt)
    out_text= model.forward(text_pt)
    cos_score = torch.cosine_similarity(out_q, out_text, dim=0)
    cos_score = torch.full_like(cos_score, 0.5) + 0.5 * cos_score
    print('score: {}'.format(cos_score.item()))

def get_symbol_text(in_file):
    symbol_doc = OrderedDict()
    fin = pd.read_csv(in_file)
    for idx, row in fin.iterrows():
        symbol = str(row['symbol_id'])
        sent = str(row['main_operation_business'])+str(row['operating_scope'])+str(row['org_cn_introduction'])
        symbol_doc[symbol] =sent
    return symbol_doc



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
    parser.add_argument('-dataset',     default='./data/DSSM_s2t/result/SH600519.csv')
    args = parser.parse_args()

    model_path = './outputs/epoch3.pt'
    file_out = './outputs/result.csv'
    model = load_model(args,model_path)
    data, query_doc = load_dataset(args)
    # get_scores(args, data, query_doc, model, file_out)
    query = 'YJ'
    text = '云集APP上买到海鲜被放到丰巢快递柜30小时 这么倒霉的事情到底怪谁。要说，云集APP既然在登录时候表示能会自动收集用户信息，这次事件就是对于用户货件的监控物流未做到位；二是云集APP在邮寄时并未贴生鲜面单，这给了快递人员错误的信息；要我说快递最委屈，照章办事却惹祸上身，冯女士也倒霉，发短信过去没看到，最终结果这个样子。'
    text2 = '教育大家云集 共议教育治理体系和治理能力现代化.中国教育报-中国教育新闻网10月24日浙江杭州讯（记者 时晓玲）何为大学治理？校长的专业发展应在哪些方面着力？在10月24日举办的浙江外国语学院教育治理研究中心成立仪式及《回望：大学校长口述》新书首发式、加快推进教育治理体系和治理能力现代化高峰论坛上，教育专家就此主题展开讨论。'
    predict_one(args, query, text, query_doc, model)
    predict_one(args, query, text2, query_doc, model)