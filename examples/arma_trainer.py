import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import ARMAConv
from data import DataLoader, DataProcessor
from utils import seed_everything, get_config, init_params, expRun, RunTimes

import matplotlib.pyplot as plt
import random
import math
import numpy as np


class ARMANet(nn.Module):
    def __init__(self, num_features, num_classes, stack_num, stack_layer_num, activation, bias, laplacian_matrix, num_layer, hidden_dim, dprate):
        super(ARMANet, self).__init__()
        self.dprate = dprate
        self.stack_num = stack_num
        self.stack_layer_num = stack_layer_num
        self.feat_encoder = nn.Linear(num_features, hidden_dim)
        self.final_encoder = nn.Linear(hidden_dim, num_classes)

        self.layers = nn.ModuleList(
            [ARMAConv(stack_num, stack_layer_num, laplacian_matrix, activation, bias, dprate, hidden_dim, hidden_dim) for i in range(num_layer)])

    def forward(self, x):
        x = F.dropout(x, self.dprate, training=self.training)
        x = self.feat_encoder(x)

        for conv in self.layers:
            x = F.dropout(x, self.dprate, training=self.training)
            x = F.relu(conv(x))

        x = self.final_encoder(x)

        return x

    def conv_visualize(self, in_x, laplacian_matrix, laplacian_type, data_name):
        laplacian_matrix = laplacian_matrix.cpu().numpy()
        pre_file = DataProcessor.get_pre_filename(laplacian_type, -0.5, False)
        eigenvals, eigenvecs =  DataProcessor.get_eigh(laplacian_matrix, False, data_name, "eigh", pre_file)

        cols = 2
        rols = math.ceil(self.stack_num/cols)            
        fig = plt.figure(figsize=(20, 9 * rols))
        xs = 1 - eigenvals

        in_x = self.feat_encoder(in_x)
        
        for idx,layer in enumerate(self.layers):
            stack_conv_vals = layer.conv_val(in_x, eigenvecs)
            in_x = layer(in_x)

            ax = fig.add_subplot(rols, cols, idx+1, projection='3d')

            stack_axis = np.array(range(1,self.stack_num+1,1))                
            for k in stack_axis:
                ys = stack_conv_vals[k-1].detach().cpu().numpy()
                ys = np.where(np.abs(ys)>500, 500, ys)
                color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
                ax.plot(xs, ys, zs=k, zdir='y', color=color, alpha=0.8)
                            
            ax.set_xlabel('frequency', size=17)
            ax.set_ylabel('frequency response', size=17)
            ax.set_title("layer: "+str(idx+1), size=20)

        plt.show()



def RunOnce(seed, data, config, device):

    seed_everything(seed)

    # process data: get laplacian, eigenvalues, eigenvectors, train/validate/test mask
    data_process_config = config.data_process
    model_config = config.model
    train_config = config.train

    data = DataProcessor(data, config.train_rate, config.val_rate, data_process_config)
    data = data.to(device)

    model = ARMANet(data.num_features, data.num_classes, model_config.stack_num, model_config.stack_layer_num, nn.ReLU(), model_config.bias, data.laplacian_matrix, model_config.num_layer,
                 model_config.hidden_dim, train_config.dropout).to(device)
    model.apply(init_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    val_acc, test_acc, time_run = expRun(model, optimizer, data, train_config.epochs, train_config.early_stopping)

    # model.conv_visualize(data.x, data.laplacian_matrix, data_process_config.laplacian_type, config.dataset)

    return val_acc, test_acc, time_run


config = get_config("ARMA")
device = torch.device('cuda:'+str(config.device)
                      if torch.cuda.is_available() else 'cpu')
# load data
data = DataLoader(config.dataset)

SEEDS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# RunOnce(0, data, config, device)
RunTimes(SEEDS, RunOnce, data, config, device)
