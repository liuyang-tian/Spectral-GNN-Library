import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import AdaConv
from data import DataLoader, DataProcessor
from utils import seed_everything, get_config, init_params, expRun, RunTimes

import matplotlib.pyplot as plt
import numpy as np
import random

class AdaGNN(nn.Module):
    def __init__(self, laplacian_matrix, num_features, num_classes, hidden_dim, num_layer, dropout):
        super(AdaGNN, self).__init__()
        self.dropout = dropout
        self.first_layer_encoder = nn.Linear(num_features, hidden_dim)
        self.output_layer_encoder = nn.Linear(hidden_dim, num_classes)

        self.layers = nn.ModuleList(
            [AdaConv(laplacian_matrix, num_features)]+[AdaConv(laplacian_matrix, hidden_dim) for i in range(num_layer-1)])

        
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[0](x)
        x = F.relu(self.first_layer_encoder(x))

        for conv in self.layers[1:]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)

        x = self.output_layer_encoder(x)
        return x

    def conv_visualize(self, laplacian_matrix, laplacian_type, self_loop, epsilon, data_name):
        laplacian_matrix = laplacian_matrix.cpu().numpy()
        pre_file = DataProcessor.get_pre_filename(laplacian_type, epsilon, self_loop)
        eigenvals, _ =  DataProcessor.get_eigh(laplacian_matrix, False, data_name, "eigh", pre_file)
        
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        conv_vals = self.layers[0].conv_val(eigenvals)
        input_channel = len(conv_vals)
        xs = eigenvals
        channel_axis = np.array(range(1,input_channel+1,1))
        for i in channel_axis:
            ys = conv_vals[i-1]
            color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
            ax.plot(xs, ys, zs=i, zdir='y', color=color, alpha=0.8)            
        ax.set_xlabel('frequency',size=20)
        ax.set_ylabel('feature chanel',size=20)
        ax.set_zlabel('frequency response',size=20)
        plt.title("layer 0",loc="center",size=20)
        plt.show()

        # layer 1 ~ n
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        
        conv_vals = 1.
        for layer in self.layers[1:]:
            conv_val = layer.conv_val(eigenvals)
            conv_val = np.vstack(conv_val)
            conv_vals = conv_vals * conv_val

        input_channel = conv_vals.shape[0]
        xs = eigenvals
        channel_axis = np.array(range(1,input_channel+1,1))
        for i in channel_axis:
            ys = conv_vals[i-1, :]
            color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
            ax.plot(xs, ys, zs=i, zdir='y', color=color, alpha=0.8)            
        ax.set_xlabel('frequency',size=20)
        ax.set_ylabel('feature chanel',size=20)
        ax.set_zlabel('frequency response',size=20)
        plt.title("layer 1~"+str(input_channel),loc="center", size=20)
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
    model = AdaGNN(data.laplacian_matrix, data.num_features, data.num_classes, model_config.hidden_dim, model_config.num_layer, train_config.dropout).to(device)
    model.apply(init_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)


    l1_params_list = ["filter_param"]
    l2_params_list = ["filter_param", "weight"]

    val_acc, test_acc, time_run = expRun(model, optimizer, data, train_config.epochs, train_config.early_stopping, True, train_config.l1norm, train_config.l2norm, l1_params_list, l2_params_list)

    # model.conv_visualize(data.laplacian_matrix, data_process_config.laplacian_type, data_process_config.add_self_loop, data_process_config.epsilon, data.name)

    return val_acc, test_acc, time_run



config = get_config("AdaGNN")
device = torch.device('cuda:'+str(config.device)
                    if torch.cuda.is_available() else 'cpu')
# load data
data = DataLoader(config.dataset)

SEEDS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# RunOnce(0, data, config, device)
RunTimes(SEEDS, RunOnce, data, config, device)
