import os
import sys

sys.path.insert(0, os.path.abspath("../"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import CorrelationFreeConv
from data import DataLoader, DataProcessor
from utils import seed_everything, get_config, init_params, expRun, RunTimes

import math
import matplotlib.pyplot as plt
import random
import numpy as np


class CorrelationFreeNet(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        k,
        basis,
        laplacian_matrix,
        eigenvals,
        eigenvecs,
        epsilon,
        shift_ratio,
        num_layer,
        hidden_dim,
        dropout,
    ):
        super(CorrelationFreeNet, self).__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.k = k
        self.feat_encoder = nn.Linear(num_features, hidden_dim)
        self.final_encoder = nn.Linear(hidden_dim, num_classes)

        self.layers = nn.ModuleList(
            [CorrelationFreeConv(k, hidden_dim) for i in range(num_layer)]
        )

        n_nodes = laplacian_matrix.shape[0]
        identity = torch.eye(n_nodes, dtype=laplacian_matrix.dtype).to(
            laplacian_matrix.device
        )

        if basis == "eps":
            filter_basis_matrix = (
                identity * (1 + shift_ratio) - laplacian_matrix
            )  # D^-epsilon @ (A+I) @ D^-epsilon

        elif basis == "rho":
            eigenval_abs = torch.abs(eigenvals)
            eigenval_abs = torch.where(
                eigenval_abs > 1e-6, eigenval_abs, torch.zeros_like(eigenval_abs)
            )
            sign = torch.ones_like(eigenvals, dtype=torch.int8)
            sign = torch.where(eigenvals >= 0, sign, -sign)
            self.filter_basis = sign * eigenval_abs.pow(epsilon)
            filter_basis_matrix = (
                eigenvecs @ self.filter_basis.diag_embed() @ eigenvecs.T
            )

        filter_bases = [identity]
        graph_matrix_n = identity

        for i in range(k):
            graph_matrix_n = torch.matmul(graph_matrix_n, filter_basis_matrix)
            filter_bases = filter_bases + [graph_matrix_n]

        self.filter_bases = torch.stack(filter_bases, dim=0).contiguous()

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=True)
        x = self.feat_encoder(x)

        for conv in self.layers:
            x = F.dropout(x, self.dropout, training=True)
            x = F.relu(conv(self.filter_bases, x))

        x = self.final_encoder(x)

        return x

    def conv_visualize(
        self,
        basis,
        laplacian_matrix,
        laplacian_type,
        epsilon,
        add_self_loop,
        shift_ratio,
        data_name,
    ):
        laplacian_matrix = (
            laplacian_matrix.cpu().numpy()
        )  # I - D^{-1/2} (A+I) D^{-1/2} / A
        epsilon = -0.5 if laplacian_type=="L0" or laplacian_type=="L1" else epsilon
        pre_file = DataProcessor.get_pre_filename(
            laplacian_type, epsilon, add_self_loop
        )
        eigenvals, _ = DataProcessor.get_eigh(
            laplacian_matrix, False, data_name, "eigh", pre_file
        )

        conv_param_list = []
        for idx, layer in enumerate(self.layers):
            conv_param = layer.conv_val()  # k * d_in
            conv_param_list.append(conv_param)

        if basis == "eps":
            filter_basis = 1 + shift_ratio - eigenvals
        elif basis == "rho":
            filter_basis = self.filter_basis.cpu().numpy()

        poly_item = np.ones_like(eigenvals)
        filter_poly = [poly_item]
        for k in range(self.k):
            poly_item = poly_item * filter_basis
            filter_poly.append(poly_item)

        fig = plt.figure(figsize=(20, 25))
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        input_channel = self.hidden_dim
        xs = eigenvals

        channel_axis = np.array(range(1, input_channel + 1, 1))
        for i in channel_axis:
            conv_val = 1.0
            for layer in range(len(conv_param_list)):
                layer_conv_val = 0.0
                for k in range(self.k + 1):
                    layer_conv_val += conv_param_list[layer][k, i - 1] * filter_poly[k]
                conv_val = conv_val * layer_conv_val
            ys = conv_val
            color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
            ax.plot(xs, ys, zs=i, zdir="y", color=color, alpha=0.8)
        ax.set_xlabel("frequency", size=20)
        ax.set_ylabel("feature chanel", size=20)
        ax.set_zlabel("frequency response", size=20)
        plt.show()

        # for i in range(input_channel):

        #     ax = fig.add_subplot(input_channel, 1, 1+i)
        #     xs =  eigenvals

        #     poly_item = np.ones_like(eigenvals)
        #     filter_poly = [poly_item]
        #     for k in range(self.k):
        #         poly_item = poly_item * filter_basis
        #         filter_poly.append(poly_item)

        #     conv_val = 1.
        #     for layer in range(len(conv_param_list)):
        #         layer_conv_val = 0.
        #         for k in range(self.k+1):
        #             layer_conv_val += conv_param_list[layer][k, i] * filter_poly[k]
        #         conv_val = conv_val * layer_conv_val

        #     color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
        #     ax.plot(xs, conv_val, color=color, alpha=0.8)

        #     ax.set_xlabel('frequency')
        #     ax.set_ylabel('feature chanel')
        #     plt.title("dim: "+str(i+1),loc="center")

        plt.show()


def RunOnce(seed, data, config, device):

    seed_everything(seed)

    # process data: get laplacian, eigenvalues, eigenvectors, train/validate/test mask
    data_process_config = config.data_process
    model_config = config.model
    train_config = config.train
    shift_ratio = model_config.shift_ratio if "shift_ratio" in model_config else 0

    data = DataProcessor(data, config.train_rate, config.val_rate, data_process_config)
    data = data.to(device)

    # init model
    model = CorrelationFreeNet(
        data.num_features,
        data.num_classes,
        model_config.k,
        data_process_config.basis,
        data.laplacian_matrix,
        data.eigenvalues,
        data.eigenvectors,
        data_process_config.epsilon,
        shift_ratio,
        model_config.num_layer,
        model_config.hidden_dim,
        train_config.dropout,
    ).to(device)
    model.apply(init_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    val_acc, test_acc, time_run = expRun(
        model, optimizer, data, config.train.epochs, config.train.early_stopping
    )

    # model.conv_visualize(
    #     data_process_config.basis,
    #     data.laplacian_matrix,
    #     data_process_config.laplacian_type,
    #     data_process_config.epsilon,
    #     data_process_config.add_self_loop,
    #     shift_ratio,
    #     config.dataset,
    # )

    return val_acc, test_acc, time_run


config = get_config("CorrelationFree")
device = torch.device(
    "cuda:" + str(config.device) if torch.cuda.is_available() else "cpu"
)

# load data
data = DataLoader(config.dataset)

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# RunOnce(0, data, config, device)
RunTimes(SEEDS, RunOnce, data, config, device)
