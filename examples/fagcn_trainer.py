import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from conv import FALayer
from data import DataLoader, DataProcessor
from utils import seed_everything, get_config, init_params, expRun, RunTimes

import matplotlib.pyplot as plt
import random


class FAGCN(nn.Module):
    def __init__(self, g, eps, num_features, num_classes, hidden_dim, num_layer, dprate, dropout):
        super(FAGCN, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = num_layer
        self.dropout = dropout
        self.dprate = dprate

        self.feat_encoder = nn.Linear(num_features, hidden_dim)
        self.final_encoder = nn.Linear(hidden_dim, num_classes)

        self.layers = nn.ModuleList(
            [FALayer(self.g, hidden_dim, dprate) for i in range(num_layer)])
    
    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.relu(self.feat_encoder(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        raw = x
        
        for i in range(self.layer_num):
            x = self.layers[i](x)
            x = self.eps * raw + x
        x = self.final_encoder(x)

        return x
    
    

def RunOnce(seed, data, config, device):

    seed_everything(seed)

    # process data: get laplacian, eigenvalues, eigenvectors, train/validate/test mask
    data_process_config = config.data_process
    model_config = config.model
    train_config = config.train

    g = data.g
    data = DataProcessor(data, config.train_rate, config.val_rate, data_process_config)
    data = data.to(device)


    # init model
    model = FAGCN(g, model_config.eps, data.num_features, data.num_classes,
                    model_config.hidden_dim, model_config.num_layer, train_config.dprate, train_config.dropout).to(device)
    model.apply(init_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    val_acc, test_acc, time_run = expRun(model, optimizer, data, train_config.epochs, train_config.early_stopping)

    
    return val_acc, test_acc, time_run


config = get_config("FAGCN")
device = torch.device('cuda:'+str(config.device)
                      if torch.cuda.is_available() else 'cpu')

# load data
data = DataLoader(config.dataset)
U = data.edge_index[0]
V = data.edge_index[1]
g = dgl.graph((U, V))
g = dgl.to_simple(g)
g = dgl.remove_self_loop(g)
g = dgl.to_bidirected(g)

g = g.to(device)
deg = g.in_degrees().float().to(device)
norm = torch.pow(deg, -0.5)
norm[torch.isinf(norm)] = 0.

g.ndata['d'] = norm

data.g = g


SEEDS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# RunOnce(1, data, config, device)
RunTimes(SEEDS, RunOnce, data, config, device)