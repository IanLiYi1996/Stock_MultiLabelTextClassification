  
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification

class Bert_MLTC(BertForSequenceClassification):
    def __init__(self):
        super(Bert_MLTC, self).__init__()

    def forward(self,x):
        return x

if __name__ == "__main__":
    pass