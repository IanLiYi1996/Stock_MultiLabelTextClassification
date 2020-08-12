from torch.util.data import DataSet, DataLoader

class trainDataSet(DataSet):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)