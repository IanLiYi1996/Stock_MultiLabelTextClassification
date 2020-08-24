import sys, os
sys.path.append('./')

from collections import OrderedDict
from ordered_set import OrderedSet
import pandas as pd, numpy as np
import tqdm

def get_stock_embed(stock_name,stock_set):
        stock_id = 0
        stock_emb = np.load("/mnt/c/Users/liyi_intern/Documents/Codes/BERT_MLTC/output/stock.npy")
        for idx, stock in enumerate(stock_set):
            if stock == stock_name:
                stock_id = idx
        return stock_emb[stock_id]

if __name__ == "__main__":
    stock = OrderedDict()
    stock_set = OrderedSet()
    df = pd.read_csv('./data/stock_info.csv', usecols=['name'])
    for index, row in tqdm.tqdm(df.iterrows()):
        name = row['name']
        if name not in stock.keys():
            stock[name]=0
        stock[name]+=1
    for key, value in stock.items():
        stock_set.add(key)
        # if value > 1:
        #     print(key, value)
    print(len(stock))
    print(len(stock_set))
    name = '中国中免'
    embd = get_stock_embed(name, stock_set)
    print(embd)