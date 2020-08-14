import pandas as pd
import numpy as np
from torch.util.data import DataSet, DataLoader

class MyDataSet(DataSet):
    """
	Training Dataset class.

	Parameters
	----------
	rootdir
    mode
	
	Returns
	-------
	A training Dataset class instance used by DataLoader
	"""
    def __init__(self, rootdir, mode):
        super(MyDataset,self).__init__()
         # TODO
        # 1. 初始化文件路径或文件名列表。
        #也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
        pass

    def __getitem__(self, index):
        # TODO

        #1.从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
        #2.预处理数据（例如torchvision.Transform）。
        #3.返回数据对（例如图像和标签）。
        #这里需要注意的是，第一步：read one data，是一个data
        pass

    def __len__(self):
        pass

class TrainDataset(Dataset):
	"""
	Training Dataset class.

	Parameters
	----------
	triples:	The triples used for training the model
	params:		Parameters for the experiments
	
	Returns
	-------
	A training Dataset class instance used by DataLoader
	"""
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params
		self.entities	= np.arange(self.p.num_ent, dtype=np.int32)

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele			= self.triples[idx]
		triple, label, sub_samp	= torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
		trp_label		= self.get_label(label)

		if self.p.lbl_smooth != 0.0:
			trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0/self.p.num_ent)

		return triple, trp_label, None, None

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, trp_label
	
	def get_neg_ent(self, triple, label):
		def get(triple, label):
			pos_obj		= label
			mask		= np.ones([self.p.num_ent], dtype=np.bool)
			mask[label]	= 0
			neg_ent		= np.int32(np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
			neg_ent		= np.concatenate((pos_obj.reshape([-1]), neg_ent))

			return neg_ent

		neg_ent = get(triple, label)
		return neg_ent

	def get_label(self, label):
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)