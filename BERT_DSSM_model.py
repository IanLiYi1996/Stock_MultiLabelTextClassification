  
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
# from torchsummary import summary
#https://pypi.douban.com/simple --load pip resource

class Bert_MLTC(nn.Module):
    def __init__(self, args):
        super(Bert_MLTC, self).__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(args.pretrained_model)
        for p in self.parameters():
            p.requires_grad = False
        self.linear1 = nn.Linear(args.nsize, args.hiddensize, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.activation1 = nn.LeakyReLU(0.001)
        self.linear2 = nn.Linear(args.hiddensize, args.hiddensize, bias=True)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.activation2 = nn.LeakyReLU(0.001)
        self.linear3 = nn.Linear(args.hiddensize, args.outfeatures, bias=True)
        nn.init.xavier_uniform_(self.linear3.weight)
        self.activation3 = nn.LeakyReLU(0.001)

    def forward(self, x):
        outputs, pooled = self.encoder(x)
        # print("Token wise output: {}, Pooled output: {}".format(outputs_q.shape, pooled_q[0].shape))
        result = self.linear1(pooled)
        result = self.activation1(result)
        # print("Linear1 output: {}".format(result.size()))
        result = self.linear2(result)
        result = self.activation2(result)
        # print("Linear2 output: {}".format(result.size()))
        result = self.linear3(result)
        return result
        
def loss(self,x, y):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pretrained_model',		default='./bert_model',					help='input the bert models file')
    parser.add_argument('-nsize', default=768,					help='input size')
    parser.add_argument('-hiddensize',		default=256,					help='hidden units')
    parser.add_argument('-outfeatures',		default=128,					help='output size')
    args = parser.parse_args()
    model = Bert_MLTC(args)
    # summary(model)
    # for param in model.parameters():
    #     print(type(param), param.size())
    print(model("This is an input example"))