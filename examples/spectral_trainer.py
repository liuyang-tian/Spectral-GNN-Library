import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import SpecConv
from data import DataLoader, DataProcessor
from utils import seed_everything, get_config, init_params, expRun, RunTimes

import matplotlib.pyplot as plt
import numpy as np
import random
import math

class SpectralNet(nn.Module):
    def __init__(self, num_features, num_classes, eigenvals, eigenvecs, k, num_layer, hidden_dim, dropout, dprate):
        super(SpectralNet, self).__init__()
        self.dropout = dropout
        self.dprate = dprate
        self.feat_encoder = nn.Linear(num_features, hidden_dim)
        self.final_encoder = nn.Linear(hidden_dim, num_classes)

        self.layers = nn.ModuleList(
            [SpecConv(hidden_dim, hidden_dim, k) for i in range(num_layer)])

        self.eigenvals = eigenvals
        self.v_k = eigenvecs[:, :k]
        self.k = k

    def forward(self, x):

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.feat_encoder(x)

        for conv in self.layers:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(conv(x, self.v_k))

        x = self.final_encoder(x)
        return x
    
    def conv_visualize(self):
        frequency_axis = self.eigenvals.cpu().numpy()

        for idx,layer in enumerate(self.layers):
            conv_param = layer.conv_val().detach().cpu().numpy()    # K*d1*d2
            input_channel = conv_param.shape[1]
            output_channel = conv_param.shape[2]
            cols = 2
            rols = math.ceil(output_channel/cols)
            
            fig = plt.figure(figsize=(20, rols * 9))
            
            for output_feat in range(output_channel):

                ax = fig.add_subplot(rols, cols, output_feat+1, projection='3d')

                xs =  frequency_axis[:self.k]
                channel_axis = np.array(range(1,input_channel+1,1))
                
                for i in channel_axis:
                    ys = conv_param[:, i-1 , 0]
                    color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
                    ax.plot(xs, ys, zs=i, zdir='y', color=color, alpha=0.8)            

                ax.set_xlabel('frequency', size=17)
                ax.set_ylabel('feature chanel', size=17)
                ax.set_zlabel('frequency response', size=17)
                plt.title("layer: "+str(idx+1)+"; output_channel: "+str(output_feat+1),loc="center", size=20)

            plt.show()

def RunOnce(seed, data, config, device):

    seed_everything(seed)

    # process data: get laplacian, eigenvalues, eigenvectors, train/validate/test mask
    data_process_config = config.data_process
    data = DataProcessor(data, config.train_rate, config.val_rate, data_process_config)
    data = data.to(device)

    # init model
    model = SpectralNet(data.num_features, data.num_classes, data.eigenvalues, data.eigenvectors, config.model.k, config.model.num_layer,
                        config.model.hidden_dim, config.train.dropout, config.train.dprate).to(device)

    model.apply(init_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    val_acc, test_acc, time_run = expRun(model, optimizer, data, config.train.epochs, config.train.early_stopping)
    # model.conv_visualize()
    return val_acc, test_acc, time_run





config = get_config("SpectralNet")
device = torch.device('cuda:'+str(config.device)
                    if torch.cuda.is_available() else 'cpu')
# load data
data = DataLoader(config.dataset)

SEEDS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# RunOnce(0, data, config, device)
RunTimes(SEEDS, RunOnce, data, config, device)
