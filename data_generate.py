#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tqdm
import pandas as pd
import codecs, sys, os

def get_filename(rootdir):
    filesnames = [ f for f in os.listdir(rootdir) if os.path.isfile(os.path.join(rootdir,f)) ]
    return filesnames

def dealing_files(filename):
    pos_data = []
    neg_data = []
    dirname_pos = 'data/DSSM/pos/'
    dirname_neg = 'data/DSSM/neg/'
    dirname_news = 'data/DSSM/news/'
    pos = pd.read_csv(dirname_pos+filename)
    for idx, row in pos.iterrows():
        data = str(row['title'])+ str(row['content'])
        pos_data.append(data.replace('nan',''))
    neg = pd.read_csv(dirname_neg+filename)
    for idx, row in neg.iterrows():
        data = str(row['title'])+ str(row['content'])
        neg_data.append(data.replace('nan',''))
    news = codecs.open(dirname_news+filename, 'r')
    for line in news.readlines():
        items = line.strip().split('')
        # print(len(items))
        # break
        data = items[6]+str(items[7].split('...')[0])
        pos_data.append(data.replace('nan',''))
    out_file = codecs.open('data/DSSM/result/'+filename, 'w', encoding='utf-8')
    title = 'symbol,pos,neg,label'+'\n'
    out_file.write(title)
    for i in range(len(pos_data)):
        for j in range(len(neg)):
            data = str(filename.replace('.csv',''))+','+pos_data[i]+','+neg_data[j]+',1'+'\n'
            out_file.write(data)


if __name__ == "__main__":
    root = 'data/DSSM/pos/'
    for filename in tqdm.tqdm(get_filename(root)):
        print(filename)
        dealing_files(filename)