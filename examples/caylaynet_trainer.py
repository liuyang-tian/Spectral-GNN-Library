import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import CayleyConv
from data import DataLoader, DataProcessor
from utils import seed_everything, get_config, init_params, expRun, RunTimes
import matplotlib.pyplot as plt
import numpy as np
import random
import math


class CaylayNet(nn.Module):
    def __init__(self, k, num_jacobi_iter, laplacian_matrix, bias, num_features, num_classes, hidden_dim, num_layer, dropout, dprate):
        super(CaylayNet, self).__init__()
        self.dropout = dropout
        self.dprate = dprate
        self.feat_encoder = nn.Linear(num_features, hidden_dim)
        self.output_encoder = nn.Linear(hidden_dim, num_classes)

        self.layers = nn.ModuleList(
            [CayleyConv(k, num_jacobi_iter, laplacian_matrix, hidden_dim, hidden_dim, bias) for i in range(num_layer)])

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.feat_encoder(x)

        for conv in self.layers:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = conv(x)

        x = self.output_encoder(x)
        return x
    
    def conv_visualize(self, laplacian_matrix, laplacian_type, data_name):
        laplacian_matrix = laplacian_matrix.cpu().numpy()
        pre_file = DataProcessor.get_pre_filename(laplacian_type, -0.5, False)
        eigenvals, _ =  DataProcessor.get_eigh(laplacian_matrix, False, data_name, "eigh", pre_file)

        for idx,layer in enumerate(self.layers):
            conv_param = layer.conv_val(eigenvals)

            input_channel = output_channel = len(conv_param)
            cols = 2
            rols = math.ceil(output_channel/cols)
            
            fig = plt.figure(figsize=(20, 9 * rols))
            
            for output_feat in range(output_channel):

                ax = fig.add_subplot(rols, cols, output_feat+1, projection='3d')

                xs =  eigenvals
                channel_axis = np.array(range(1,input_channel+1,1))
                
                for i in channel_axis:
                    ys = conv_param[output_feat][i-1]
                    color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
                    ax.plot(xs, ys, zs=i, zdir='y', color=color, alpha=0.8)            

                ax.set_xlabel('frequency', size=17)
                ax.set_ylabel('feature chanel', size=17)
                ax.set_zlabel('frequency response', size=17)
                plt.title("layer: "+str(idx+1)+"; output_channel: "+str(output_feat+1),loc="center",size=20)

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
    model = CaylayNet(model_config.k, model_config.num_jacobi_iter, data.laplacian_matrix, model_config.bias, data.num_features, data.num_classes, model_config.hidden_dim, model_config.num_layer, train_config.dropout, train_config.dprate).to(device)
    model.apply(init_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    val_acc, test_acc, time_run = expRun(model, optimizer, data, train_config.epochs, train_config.early_stopping)

    # model.conv_visualize(data.laplacian_matrix, data_process_config.laplacian_type, config.dataset)

    return val_acc, test_acc, time_run



config = get_config("CayleyNet")
device = torch.device('cuda:'+str(config.device)
                      if torch.cuda.is_available() else 'cpu')

# load data
data = DataLoader(config.dataset)

SEEDS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# RunOnce(0, data, config, device)
RunTimes(SEEDS, RunOnce, data, config, device)