import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import BernConv
from data import DataLoader, DataProcessor
from utils import seed_everything, get_config, init_params, expRun, RunTimes

import matplotlib.pyplot as plt
import random


class BernNet(nn.Module):
    def __init__(self, k, laplacian_matrix, num_features, num_classes, hidden_dim, num_layer, dprate, dropout):
        super(BernNet, self).__init__()
        self.dropout = dropout
        self.dprate = dprate
        self.feat_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
            )
        self.final_encoder = nn.Linear(hidden_dim, num_classes)

        self.layers = nn.ModuleList(
            [BernConv(k, laplacian_matrix) for i in range(num_layer)])


    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.feat_encoder(x)

        for conv in self.layers:
            if self.dprate == 0.0:
                x = F.relu(conv(x))
            else:
                x = F.dropout(x, p=self.dprate, training=self.training)
                x = F.relu(conv(x))

        x = self.final_encoder(x)

        return x

    def conv_visualize(self, laplacian_matrix, laplacian_type, data_name):
        laplacian_matrix = laplacian_matrix.cpu().numpy()
        pre_file = DataProcessor.get_pre_filename(laplacian_type, -0.5, False)
        eigenvals, _ =  DataProcessor.get_eigh(laplacian_matrix, False, data_name, "eigh", pre_file)
        
        conv_param = 1.
        for layer in self.layers:
            layer_conv = layer.conv_val(eigenvals)
            conv_param = conv_param * layer_conv        
        
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(1, 1, 1)
        color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
        xs = eigenvals
        ys = conv_param
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
    model = BernNet(model_config.k, data.laplacian_matrix, data.num_features, data.num_classes,
                    model_config.hidden_dim, model_config.num_layer, train_config.dprate, train_config.dropout).to(device)
    model.apply(init_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    val_acc, test_acc, time_run = expRun(model, optimizer, data, train_config.epochs, train_config.early_stopping)

    # model.conv_visualize(data.laplacian_matrix, data_process_config.laplacian_type, config.dataset)

    return val_acc, test_acc, time_run





config = get_config("BernNet")
device = torch.device('cuda:'+str(config.device)
                      if torch.cuda.is_available() else 'cpu')

# load data
data = DataLoader(config.dataset)

SEEDS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# RunOnce(0, data, config, device)
RunTimes(SEEDS, RunOnce, data, config, device)
