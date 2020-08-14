#!/usr/bin/env python
# -*- coding: utf-8 -*-
#first run
'''
bert-serving-start -model_dir /mnt/c/Users/liyi_intern/Documents/Codes/bert_model/ -num_worker=1 -pooling_strategy NONE
'''

from bert_serving.client import BertClient
import numpy as np
import pandas as pd

bc = BertClient()
result = bc.encode(['First do it', 'then do it right', 'then do it better'])
print(result.shape)
result = bc.encode(['First do it ||| then do it right'])
print(result.shape)

# if __name__ == "__main__":
#     stock = pd.read_csv('./stock_info.csv', usecols=['name', 'main_operation_business', 'operating_scope','org_cn_introduction'])
#     for index, row in stock.iterrows():
        