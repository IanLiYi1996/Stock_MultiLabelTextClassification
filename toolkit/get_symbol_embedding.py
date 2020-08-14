#!/usr/bin/env python
# -*- coding: utf-8 -*-
#first run
'''
bert-serving-start -model_dir /mnt/c/Users/liyi_intern/Documents/Codes/bert_model/ -num_worker=1 -pooling_strategy NONE
'''

from bert_serving.client import BertClient
import numpy as np
import pandas as pd
import re,sys
sys.path.append('/mnt/c/Users/liyi_intern/Documents/Codes/BERT_MLTC')
from toolkit.clean_text import TextProcess, DataClean
procText = TextProcess()
cleaner = DataClean()

import tqdm


def getSent(row):
    listSent = []
    if str(row['main_operation_business']) is not 'nan':
        items = procText.process_text(cleaner.clean_text(str(row['main_operation_business']))).split('。')
        listSent+=items
    if str(row['operating_scope']) is not 'nan':
        items = procText.process_text(cleaner.clean_text(str(row['operating_scope']))).split('。')
        listSent+=items
    if str(row['org_cn_introduction']) is not 'nan':
        items = procText.process_text(cleaner.clean_text(str(row['org_cn_introduction']))).split('。')
        listSent+=items
    return listSent


if __name__ == "__main__":
    # sents = []
    # bc = BertClient(check_length=False)
    # result = []
    # stock = pd.read_csv('./stock_info.csv', usecols=['name', 'main_operation_business', 'operating_scope','org_cn_introduction'])
    # for index, row in tqdm.tqdm(stock.iterrows()):
    #     sents = getSent(row)
    #     while '' in sents:
    #         sents.remove('')
    #     while 'nan' in sents:
    #         sents.remove('nan')
    #     intro = '|||'.join(sents)
    #     sents.clear()
    #     sents.append(intro)
    #     emb = bc.encode(sents)
    #     emb = emb.tolist()
    #     result.append(emb[0])
    # print(np.array(result).shape)
    # np.save("stock.npy", np.array(result))
    data = np.load("/mnt/c/Users/liyi_intern/Documents/Codes/BERT_MLTC/output/stock.npy")
    print(data.shape)