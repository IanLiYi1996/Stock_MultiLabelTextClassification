  #!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
# from torchsummary import summary
#https://pypi.douban.com/simple --load pip resource

class BertDSSM(nn.Module):
    def __init__(self, args):
        super(BertDSSM, self).__init__()
        self.args = args
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
        result = self.linear1(x)
        result = self.activation1(result)
        result = self.linear2(result)
        result = self.activation2(result)
        result = self.linear3(result)
        return result


def entropy(p):
    """ Compute the entropy of a probability distribution """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pretrained_model',		default='./bert_model',					help='input the bert models file')
    parser.add_argument('-nsize', default=768,					help='input size')
    parser.add_argument('-hiddensize',		default=256,					help='hidden units')
    parser.add_argument('-outfeatures',		default=128,					help='output size')
    parser.add_argument('-max_seq_len',		default=128,					help='input the test file')
    args = parser.parse_args()
    model = BertDSSM(args)
    example = torch.randn(3, 768)
    print(model(example).size())