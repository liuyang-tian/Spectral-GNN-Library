import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import ChebyConv2
from data import DataLoader, DataProcessor
from utils import seed_everything, get_config, init_params, expRun, RunTimes

import matplotlib.pyplot as plt
import random
import numpy as np


class ChebyNet2(nn.Module):
    def __init__(self, num_features, num_classes, laplacian_matrix, eigenvals, k, num_layer, hidden_dim, dropout):
        super(ChebyNet2, self).__init__()
        self.dropout = dropout
        self.feat_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU()
        )
        self.final_encoder = nn.Linear(hidden_dim, num_classes)

        self.layers = nn.ModuleList(
            [ChebyConv2(k) for i in range(num_layer)])

        # chebynodes_vals
        cheby_nodes=[]
        for j in range(0, k+1):
            x_j = math.cos((j+0.5)*math.pi/(k+1))
            cheby_nodes.append(x_j)
        cheby_nodes_val=[]
        for x_j in cheby_nodes:
            chebynode_val=[]
            for j in range(0, k+1):
                if j == 0:
                    chebynode_val.append([1])
                elif j == 1:
                    chebynode_val.append([x_j])
                else:
                    item = 2 * x_j * chebynode_val[j-1][0] - chebynode_val[j-2][0]
                    chebynode_val.append([item])
            chebynode_val = torch.Tensor(chebynode_val)
            cheby_nodes_val.append(chebynode_val)
        self.chebynodes_vals=torch.cat(cheby_nodes_val, dim=1).to(laplacian_matrix.device)
    
        # scaled Laplacian matrix
        max_eigenvalue = max(eigenvals)
        self.scaled_laplacian = 2/max_eigenvalue * laplacian_matrix - \
            torch.eye(laplacian_matrix.shape[0]).to(laplacian_matrix.device)

    def forward(self, x):

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.feat_encoder(x)

        for conv in self.layers:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(conv(self.chebynodes_vals, self.scaled_laplacian, x))
        
        x = self.final_encoder(x)

        return x

    
    def conv_visualize(self, laplacian_matrix, laplacian_type, data_name):
        laplacian_matrix = laplacian_matrix.cpu().numpy()
        pre_file = DataProcessor.get_pre_filename(laplacian_type, -0.5, False)
        eigenvals, _ =  DataProcessor.get_eigh(laplacian_matrix, False, data_name, "eigh", pre_file)
        
        conv_param = 1.
        for layer in self.layers:
            layer_conv = layer.conv_val(self.chebynodes_vals, eigenvals)
            conv_param = conv_param * layer_conv        
        
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(1, 1, 1)
        color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
        xs = eigenvals
        ys = np.squeeze(conv_param)
        ax.plot(xs, ys, color=color, alpha=0.8)            
        ax.set_xlabel('frequency',size=20)
        ax.set_ylabel('frequency response',size=20)
        plt.show()

def RunOnce(seed, data, config, device):

    seed_everything(seed)

    # process data: get laplacian, eigenvalues, eigenvectors, train/validate/test mask
    data_process_config = config.data_process
    model_config = config.model
    train_config = config.train

    data = DataProcessor(data, config.train_rate, config.val_rate, data_process_config)
    data = data.to(device)

    # init model
    model = ChebyNet2(data.num_features, data.num_classes, data.laplacian_matrix, data.eigenvalues, model_config.k, model_config.num_layer,
                    model_config.hidden_dim, train_config.dropout,).to(device)
    model.apply(init_params)

    optimizer = torch.optim.Adam([{ 'params': model.feat_encoder.parameters(), 'weight_decay': config.train.weight_decay, 'lr': config.train.mlp_lr},
            {'params': model.final_encoder.parameters(), 'weight_decay': config.train.weight_decay, 'lr': config.train.mlp_lr},
            {'params': model.layers.parameters(), 'weight_decay': config.train.weight_decay, 'lr': config.train.lr}])

    val_acc, test_acc, time_run = expRun(model, optimizer, data, config.train.epochs, config.train.early_stopping)

    # model.conv_visualize(data.laplacian_matrix, data_process_config.laplacian_type, config.dataset)

    return val_acc, test_acc, time_run




config = get_config("ChebyNet2")
device = torch.device('cuda:'+str(config.device)
                      if torch.cuda.is_available() else 'cpu')

# load data
data = DataLoader(config.dataset)

SEEDS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# RunOnce(0, data, config, device)
RunTimes(SEEDS, RunOnce, data, config, device)