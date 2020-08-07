import logging
import os
import sys

from data import *
from model import *
import torch
import torch.nn as nn


model = Bert_MLTC()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

def train(epoch):
    for i, data in enumerate(dataLoader, 0):
        x, y = data
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    train()