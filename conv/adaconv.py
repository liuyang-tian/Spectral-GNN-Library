import os
import sys
sys.path.insert(0, os.path.abspath('../'))

from data import DataProcessor
import torch
import torch.nn as nn
import numpy as np


class AdaConv(nn.Module):
    def __init__(self, laplacian_matrix, input_dim):
        super(AdaConv, self).__init__()
        self.filter_param = nn.Parameter(torch.Tensor(input_dim))
        self.laplacian_matrix = laplacian_matrix
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.filter_param, mean=0, std=0)

    def forward(self, x):
        y = x - self.laplacian_matrix @ x * self.filter_param.unsqueeze(dim=0)
        return y
    
    def conv_val(self, eigenvals):
        conv_vals = []
        conv_param = self.filter_param.detach().cpu().numpy()
        ones = np.ones_like(eigenvals)
        for i in range(conv_param.shape[0]):
            conv_val = ones - conv_param[i] * eigenvals
            conv_vals.append(conv_val)

        return conv_vals

