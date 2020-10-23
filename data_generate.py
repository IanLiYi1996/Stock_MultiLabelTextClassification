#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tqdm
import random
import pandas as pd
import codecs, sys, os

def get_filename(rootdir):
    filesnames = [ f for f in os.listdir(rootdir) if os.path.isfile(os.path.join(rootdir,f)) ]
    return filesnames

def dealing_files(filename):
    print(filename)
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
    title = 'symbol0x10pos0x10neg0x10label'+'\n'
    out_file.write(title)
    assert(len(pos_data)>0)
    assert(len(neg_data)>0)
    for i in range(len(pos_data)):
        j = random.randint(0, len(neg_data)-1)
        data = str(filename.replace('.csv',''))+'0x10'+pos_data[i]+'0x10'+neg_data[j]+'0x10'+'1'+'\n'
        out_file.write(data)

def get_dataset(filename, out_file):
    print(filename)
    df = pd.read_csv(filename, sep='0x10')
    fout = codecs.open(out_file, 'a+', encoding='utf-8')
    for i in range(0,130):
        loc = random.randint(0, df.shape[0]-1)
        data = df.iloc[loc].tolist()
        out = str(data[0])
        for j in range(1,len(data)):
            out = out+'0x10'+str(data[j])
        out = out+'\n'
        fout.write(out)


def split_train_test(data):
    data_all = []
    fin = codecs.open(data, 'r', encoding='utf-8')
    ftrian = codecs.open('data/DSSM/train.csv', 'w', encoding='utf-8')
    ftest = codecs.open('data/DSSM/test.csv', 'w', encoding='utf-8')
    for line in fin.readlines():
        data_all.append(line.strip())
    random.shuffle(data_all)
    for i in range(0, int(len(data_all)*0.7)):
        ftrian.write(str(data_all[i])+'\n')
    for i in range(int(len(data_all)*0.7)+1, len(data_all)):
        ftest.write(str(data_all[i])+'\n')


if __name__ == "__main__":
    root = 'data/DSSM/result/'
    output = 'data/DSSM/select.csv'
    split_train_test(output)
    # for filename in tqdm.tqdm(get_filename(root)):
    #     get_dataset(root+filename, output)
    #     # dealing_files(filename)