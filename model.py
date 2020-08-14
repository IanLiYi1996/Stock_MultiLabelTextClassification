  
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
        self.linear1 = nn.Linear(args.nsize, args.outfeatures, bias=False)

    def forward(self,x):
        tokens = self.tokenizer.tokenize(x)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_ids)
        tokens_pt = torch.tensor([tokens_ids])
        outputs, pooled = self.encoder(tokens_pt)
        result = self.linear1(pooled[0])
        return result 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pretrained_model',		default='./bert_model',					help='input the bert models file')
    parser.add_argument('-nsize',		default=768,					help='input size')
    parser.add_argument('-outfeatures',		default=200,					help='output size')
    args = parser.parse_args()
    model = Bert_MLTC(args)
    print(model("This is an input example"))