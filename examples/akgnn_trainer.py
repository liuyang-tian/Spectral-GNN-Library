import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import AKGNNConv
from data import DataLoader, DataProcessor
from utils import seed_everything, get_config, init_params, expRun, RunTimes

import matplotlib.pyplot as plt
import random
import math


class AKGNN(nn.Module):
    def __init__(self, edge_index, num_features, num_classes, hidden_dim, num_layer, dprate):
        super(AKGNN, self).__init__()
        self.dprate = dprate
        self.num_layer = num_layer
        self.feat_encoder = nn.Linear(num_features, hidden_dim)
        self.final_encoder = nn.Linear(hidden_dim * num_layer, num_classes)

        self.layers = nn.ModuleList(
            [AKGNNConv(edge_index) for i in range(num_layer)])

    def forward(self, x):
        x = F.dropout(x, self.dprate, training=self.training)
        x = F.leaky_relu(self.feat_encoder(x))

        layer_list = []        
        for conv in self.layers:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = conv(x)
            layer_list.append(x)
        
        x = torch.cat(layer_list, dim=1)   # N * (k*d)
        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.final_encoder(x)
        return x

    def conv_visualize(self, adj, data_name):
        adj = adj.cpu().numpy()
        laplacian_matrix = DataProcessor.get_Laplacian(adj, "L2", True)
        pre_file = DataProcessor.get_pre_filename("L2", -0.5, True)
        eigenvals, _ =  DataProcessor.get_eigh(laplacian_matrix, False, data_name, "eigh", pre_file)

        cols = 2
        rols = math.ceil(self.num_layer / cols)
        fig = plt.figure(figsize=(20, 9 * rols))

        conv_param = 1.
        for idx, layer in enumerate(self.layers):
            ax = fig.add_subplot(rols, cols, idx+1)
            alpha1, alpha2, D_mean = layer.conv_val(adj)
            layer_filter = (alpha1 - alpha2) / D_mean + alpha2 * (1 - eigenvals)
            conv_param = conv_param * layer_filter        
        
            color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
            xs = eigenvals
            ys = conv_param
            ax.plot(xs, ys, color=color, alpha=0.8)

        ax.set_xlabel('frequency',size=17)
        ax.set_ylabel('frequency response',size=17)
        ax.set_title('layer: '+ str(idx+1), size=20)
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
    model = AKGNN(data.edge_index, data.num_features, data.num_classes,
                model_config.hidden_dim, model_config.num_layer, train_config.dprate).to(device)
    model.apply(init_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    val_acc, test_acc, time_run = expRun(model, optimizer, data, train_config.epochs, train_config.early_stopping)

    # model.conv_visualize(data.laplacian_matrix, config.dataset)

    return val_acc, test_acc, time_run


config = get_config("AKGNN")
device = torch.device('cuda:'+str(config.device)
                    if torch.cuda.is_available() else 'cpu')
# load data
data = DataLoader(config.dataset)

SEEDS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# RunOnce(0, data, config, device)
RunTimes(SEEDS, RunOnce, data, config, device)

