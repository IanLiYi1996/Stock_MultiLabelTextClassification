import pandas as pd
import numpy as np
from torch.util.data import DataSet, DataLoader
from transformers import AutoTokenizer
import argparse

class MyDataSet(DataSet):
    """
	Training Dataset class.

	Parameters
	----------
	triples:	The quadples(title,text,stock_name,label) used for training the model
	params:		Parameters for the experiments
	stock_dict:     Dict for stock and embedding

	Returns
	-------
	A training Dataset class instance used by DataLoader
	"""
    def __init__(self, quadples, params, stock_dict):
        super(MyDataset,self).__init__()
        self.quadples = quadples
        self.p = params
        self.stock_dict = stock_dict
        self.stock_emb = np.load(self.p.stock_emb_file)

    def __getitem__(self, index):
        sample = self.quadples[index]
        title,text,stock_name,label = sample['title'],sample['text'],sample['stock_name'],sample['label']
        tokenizer = AutoTokenizer.from_pretrained(self.p.pretrained_model)

        title_tokens = tokenizer.tokenize(title)
        title_tokens_ids = tokenizer.convert_tokens_to_ids(title_tokens_tokens)
        title_tokens_ids = tokenizer.build_inputs_with_special_tokens(title_tokens_ids)
        title_tokens_pt = torch.tensor([title_tokens_ids])

        text_tokens = tokenizer.tokenize(text)
        text_tokens_ids = tokenizer.convert_tokens_to_ids(text_tokens_tokens)
        text_tokens_ids = tokenizer.build_inputs_with_special_tokens(text_tokens_ids)
        text_tokens_pt = torch.tensor([text_tokens_ids])

        stock_embed = get_stock_embed(stock_name)

        return title_tokens_pt, text_tokens_pt, stock_embed, label

    def __len__(self):
        return len(self.quadples)

    def get_stock_embed(self, stock_name):
        stock_id = 0
        for idx, stock in enumerate(self.stock_dict):
            if stock == stock_name:
                stock_id = idx
        return self.stock_emb[stock_id]

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pretrained_model',		default='./bert_model',					help='input the bert models file')
    parser.add_argument('-nsize',		default=768,					help='input size')
    parser.add_argument('-outfeatures',		default=200,					help='output size')
    parser.add_argument('-stock_emb_file',		default='./output/stock.npy',					help='output size')
    parser.add_argument('-outfeatures',		default=200,					help='output size')
    args = parser.parse_args()
    # dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)