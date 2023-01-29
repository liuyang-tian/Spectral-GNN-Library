import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import numpy as np
import torch
import torch.nn as nn
from utils import pmi
from data import DataProcessor


class DSGCConv(nn.Module):
    def __init__(self, obj_conv_method, obj_laplacian_mat, alpha, lam, mu, repeat):
        super(DSGCConv, self).__init__()
        self.obj_conv_method = obj_conv_method
        self.obj_laplacian_mat = obj_laplacian_mat
        self.alpha = alpha
        self.lam = lam
        self.mu = mu
        self.repeat = repeat

    def forward(self, x, F):
        # Attribute Graph Convolution
        x = x @ F   # X @ F

        # Object Graph Convolution
        if self.obj_conv_method == 'LP':
            y = self.ap_approximate(self.obj_laplacian_mat, x, self.alpha)          
        else:
            y = self.taubin_smoothing(self.obj_laplacian_mat, x, self.lam, self.mu, self.repeat)

        return y

    def taubin_smoothing(self, laplacian_matrix, features, lam=1, mu=1, repeat=2):
        n = laplacian_matrix.shape[0]
        device = laplacian_matrix.device
        smoothor = torch.eye(n).to(device) * (1 - lam) + lam * laplacian_matrix
        inflator = torch.eye(n).to(device) * (1 - mu) + mu * laplacian_matrix
        step_transformor = smoothor * inflator
        for i in range(repeat):
            features = step_transformor @ features
        return features
    
    def ap_approximate(self, laplacian_matrix, features, alpha=10):
        device = laplacian_matrix.device
        k = int(np.ceil(4 / alpha))
        laplacian_matrix = laplacian_matrix / (alpha + 1)
        new_feature = torch.zeros_like(features).to(device)
        for _ in range(k):
            new_feature = laplacian_matrix @ new_feature
            new_feature += features
        new_feature *= alpha / (alpha + 1)
        return new_feature

    def conv_val(self, G_eigenvals, G_eigenvecs, input_x, F_matrix):
        if self.obj_conv_method=="2-order_L":
            smoothor = 1. * (1 - self.lam) + self.lam * G_eigenvals
            inflator = 1. * (1 - self.mu) + self.mu * G_eigenvals
            step_transformor = smoothor * inflator
            G_conv_vals = step_transformor ** self.repeat
        elif self.obj_conv_method=="LP":
            in_x = input_x @ F_matrix   # X @ F
            out_x = self.ap_approximate(self.obj_laplacian_mat, in_x, self.alpha)
            in_x = in_x.cpu().numpy()
            out_x = out_x.cpu().numpy()
            spec_out = (G_eigenvecs.T @ out_x).sum(1)
            spec_in = (G_eigenvecs.T @ in_x).sum(1)          
            G_conv_vals = in_x.shape[1] / out_x.shape[1] * ( spec_out / spec_in )
        return G_conv_vals


    

