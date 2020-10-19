  
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BertTokenizer
import argparse
#https://pypi.douban.com/simple --load pip resource

class Bert_MLTC(nn.Module):
    def __init__(self, args):
        super(Bert_MLTC, self).__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(args.pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        for p in self.parameters():
            p.requires_grad = False
        self.linear1 = nn.Linear(args.nsize, args.outfeatures, bias=True)
        self.activation1 = nn.LeakyReLU(0.001)
        self.linear2 = nn.Linear(args.nsize, args.outfeatures, bias=True)
        self.activation2 = nn.LeakyReLU(0.001)
        self.linear2 = nn.Linear(args.nsize, args.outfeatures, bias=True)
        self.activation3 = nn.LeakyReLU(0.001)

    def forward(self,q, pos=None, neg=None):
        tokens_q = self.tokenizer.tokenize(q)
        print("Tokens: {}".format(tokens_q))
        tokens_ids_q = self.tokenizer.convert_tokens_to_ids(tokens_q)
        print("Tokens id: {}".format(tokens_ids_q))
        tokens_ids_q = self.tokenizer.build_inputs_with_special_tokens(tokens_ids_q)
        tokens_pt_q = torch.tensor([tokens_ids_q])
        print("Tokens PyTorch: {}".format(tokens_pt_q))
        outputs_q, pooled_q = self.encoder(tokens_pt_q)
        print("Token wise output: {}, Pooled output: {}".format(outputs_q.shape, pooled_q[0].shape))
        result = self.linear1(pooled_q[0])
        return result
        
    def loss(self,x, y):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pretrained_model',		default='./bert_model',					help='input the bert models file')
    parser.add_argument('-nsize',		default=768,					help='input size')
    parser.add_argument('-hiddensize',		default=256,					help='hidden units')
    parser.add_argument('-outfeatures',		default=128,					help='output size')
    args = parser.parse_args()
    model = Bert_MLTC(args)
    print(model("This is an input example"))