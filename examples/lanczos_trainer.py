import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import LanczosConv
from data import DataLoader, DataProcessor
from utils import seed_everything, get_config, init_params, expRun

import matplotlib.pyplot as plt
import numpy as np
import random
import math


import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import LanczosConv
from data import DataLoader, DataProcessor
from utils import seed_everything, get_config, init_params, expRun, RunTimes

import matplotlib.pyplot as plt
import numpy as np
import random
import math


class LanczosNet(nn.Module):
    def __init__(self, k, num_features, num_classes, laplacian_matrix, eigenvalues, eigenvectors, long_diffusion_list, short_diffusion_list, mlp_dim, num_layer, hidden_dim, dropout):
        super(LanczosNet, self).__init__()
        self.dropout = dropout
        self.feat_encoder = nn.Linear(num_features, hidden_dim)
        self.final_encoder = nn.Linear(hidden_dim, num_classes)

        self.long_diffusion_list = long_diffusion_list
        self.short_diffusion_list = short_diffusion_list

        self.layers = nn.ModuleList([LanczosConv(
            len(self.long_diffusion_list), len(self.short_diffusion_list), mlp_dim, hidden_dim) for i in range(num_layer)])

        self.laplacian_matrix = laplacian_matrix
        
        # magnitude
        eigs_M = torch.abs(eigenvalues)
        # sort it following descending order
        idx = torch.argsort(eigs_M ,descending=True)
        self.eigenvalues = eigenvalues[idx[:k]]
        self.eigenvectors = eigenvectors[:, idx[:k]]
        self.idx = idx[:k]

        # (eigen_vals) ^ [long_diffusion_list]
        eigen_vals = self.eigenvalues.view(-1, 1)
        eigen_vals_long_pow = []
        for I_i in self.long_diffusion_list:
            eigv_pow = torch.pow(eigen_vals, I_i)
            eigen_vals_long_pow.append(eigv_pow)
        self.eigen_vals_long_pow = torch.cat(
            eigen_vals_long_pow, dim=1)   # k*len(long_diffusion_list)

    def forward(self, x):

        x = F.dropout(x, self.dropout, training=True)
        x = self.feat_encoder(x)

        for conv in self.layers:
            x = F.dropout(x, self.dropout, training=True)
            # [L^s1 * X, L^s2 * X,...L^sm * X, L1*X, L2*X...Le*X] * W
            x = F.relu(conv(self.laplacian_matrix, self.eigenvectors, x,
                       self.short_diffusion_list, self.eigen_vals_long_pow))  # N*d2

        x = self.final_encoder(x)
        return x

    def conv_visualize(self, laplacian_matrix, laplacian_type, data_name):
        pre_file = DataProcessor.get_pre_filename(laplacian_type, -0.5, True)
        all_eigenvals, _ = DataProcessor.get_eigh(laplacian_matrix.cpu().numpy(), False, data_name, "eigh", pre_file)
        short_len = len(self.short_diffusion_list)

        for idx,layer in enumerate(self.layers):
            long_filter, weight_matrix = layer.conv_val(self.eigen_vals_long_pow)
            long_filter = long_filter.T   # l*k 
            
            input_channel = output_channel = weight_matrix.shape[0]
            cols = 2
            rols = math.ceil(output_channel/cols)
            
            fig = plt.figure(figsize=(20, 9*rols))
            
            for output_feat in range(output_channel):

                ax = fig.add_subplot(rols, cols, output_feat+1, projection='3d')

                xs =  1 - all_eigenvals
                channel_axis = np.array(range(1,input_channel+1,1))
                
                for i in channel_axis:
                    l_conv_val = 0.
                    for k in range(long_filter.shape[0]):
                        l_conv_val += weight_matrix[output_feat, (short_len+k)*input_channel+i-1] * long_filter[k,:]
                    conv_val = np.zeros_like(all_eigenvals)
                    conv_val[self.idx.cpu().numpy()] = l_conv_val
                    for si, s in enumerate(self.short_diffusion_list):
                        conv_val += weight_matrix[output_feat, si*input_channel+i-1] * all_eigenvals**s
                    ys = conv_val
                    color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
                    ax.plot(xs, ys, zs=i, zdir='y', color=color, alpha=0.8)            
                ax.set_xlabel('frequency', size = 17)
                ax.set_ylabel('feature chanel', size = 17)
                ax.set_zlabel('frequency response',size = 17)
                plt.title("layer: "+str(idx+1)+"; output_channel: "+str(output_feat+1),loc="center", size = 20)

            plt.show()

def RunOnce(seed, data, config, device):
    seed_everything(seed)
    
    data_process_config = config.data_process
    model_config = config.model
    train_config = config.train

    data = DataProcessor(data, config.train_rate, config.val_rate, data_process_config)
    data = data.to(device)

    # init model
    model = LanczosNet(data_process_config.k, data.num_features, data.num_classes, data.laplacian_matrix, data.eigenvalues, data.eigenvectors, model_config.long_diffusion_list, model_config.short_diffusion_list, model_config.mlp_dim, model_config.num_layer,
                    model_config.hidden_dim, train_config.dropout).to(device)
    model.apply(init_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
        
    val_acc, test_acc, time_run = expRun(model, optimizer, data, config.train.epochs, config.train.early_stopping)

    # model.conv_visualize(data.laplacian_matrix, data_process_config.laplacian_type, config.dataset)

    return val_acc, test_acc, time_run


config = get_config("LanczosNet")
device = torch.device('cuda:'+str(config.device)
                      if torch.cuda.is_available() else 'cpu')


# load data
data = DataLoader(config.dataset)

SEEDS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# RunOnce(0, data, config, device)
RunTimes(SEEDS, RunOnce, data, config, device)








