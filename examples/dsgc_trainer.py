import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import DSGCConv
from data import DataLoader, DataProcessor
from utils import seed_everything, get_config, init_params, expRun, construct_pmi_matrix, RunTimes

import numpy as np
import matplotlib.pyplot as plt

class DSGC(nn.Module):
    def __init__(self, features, attri_laplacian_type, obj_conv_method, obj_laplacian_mat, alpha, lam, mu, repeat, num_features, num_classes, hidden_dim, dprate, dataset):
        super(DSGC, self).__init__()
        self.dprate = dprate

        self.conv = DSGCConv(obj_conv_method, obj_laplacian_mat, alpha, lam, mu, repeat)
        self.final_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
            )
        
        # construct attribute affinity graph (F)
        pmi_matrix = construct_pmi_matrix(features, dataset)
        self.F_adj = DataProcessor.get_Laplacian(adj=pmi_matrix, laplacian_type=attri_laplacian_type, add_self_loop=True)
        device = features.device
        self.F = 0.5 * (torch.eye(self.F_adj.shape[0]) + torch.Tensor(self.F_adj)).to(device)


    def forward(self, x):
        x = F.dropout(x, self.dprate, training=True)
        x = F.relu(self.conv(x, self.F))

        x = self.final_encoder(x)

        return x

    def conv_visualize(self, obj_laplacian_mat, attri_laplacian_type, obj_laplacian_type, epsilon, input_feat, data_name):
        F_matrix = self.F_adj    # D^{-1/2} (A+I) D^{-1/2} / A
        F_pre_file = DataProcessor.get_pre_filename(attri_laplacian_type, -0.5, True)
        F_eigenvals, _ =  DataProcessor.get_eigh(F_matrix, False, data_name, "pmi", F_pre_file)

        G_matrix = obj_laplacian_mat.cpu().numpy()
        G_pre_file = DataProcessor.get_pre_filename(obj_laplacian_type, epsilon, True)
        G_eigenvals, G_eigenvecs =  DataProcessor.get_eigh(G_matrix, False, data_name, "eigh", G_pre_file)

        F_conv = 1 / 2 * (1 + F_eigenvals)
        G_conv = self.conv.conv_val(G_eigenvals, G_eigenvecs, input_feat, self.F)
        # conv_vals = np.array([[i * j for i in G_conv] for j in F_conv])

        x = (1 - G_eigenvals).reshape(1,-1)
        y = (1 - F_eigenvals).reshape(-1,1)
        z = F_conv.reshape(-1,1) @ G_conv.reshape(1,-1)
        
        fig = plt.figure(figsize=(20,15))
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
        ax.set_xlabel("object_frequency", size=20)
        ax.set_ylabel("attribute_frequency", size=20)
        ax.set_zlabel("frequency response", size=20)
        plt.show()


def RunOnce(seed, data, config, device):
    seed_everything(seed)
    
    data_process_config = config.data_process
    model_config = config.model
    train_config = config.train

    obj_conv_method = "2-order_L"
    if any([data.name == dname for dname in ['chameleon', 'film', 'squirrel', 'texas', 'cornell']]):
        data_process_config.laplacian_type = 'L3'
        data_process_config.epsilon = -0.5
        obj_conv_method = "LP"

    # process data: get laplacian, eigenvalues, eigenvectors, train/validate/test mask
    data = DataProcessor(data, config.train_rate,
                        config.val_rate, data_process_config)

    data = data.to(device)

    # init model
    model = DSGC(data.x, data_process_config.F.laplacian_type, obj_conv_method, data.laplacian_matrix, 
                data_process_config.alpha, data_process_config.taubin_lambda, data_process_config.taubin_mu, 
                data_process_config.taubin_repeat, data.num_features, data.num_classes, model_config.hidden_dim, 
                train_config.dprate, data.name).to(device)
    model.apply(init_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    val_acc, test_acc, time_run = expRun(model, optimizer, data, train_config.epochs, train_config.early_stopping)

    # model.conv_visualize(data.laplacian_matrix, data_process_config.F.laplacian_type, data_process_config.laplacian_type, data_process_config.epsilon, data.x, config.dataset)

    return val_acc, test_acc, time_run

    

    

config = get_config("DSGC")
device = torch.device('cuda:'+str(config.device)
                      if torch.cuda.is_available() else 'cpu')

# load data
data = DataLoader(config.dataset)

SEEDS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# RunOnce(0, data, config, device)
RunTimes(SEEDS, RunOnce, data, config, device)