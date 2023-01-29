import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AKGNNConv(nn.Module):
    def __init__(self, edge_index):
        super(AKGNNConv, self).__init__()
        self.edge_index = edge_index
        self.lambda_param = nn.Parameter(torch.tensor(1.))
        self.special_spmm = SpecialSpmm()
        self.i = None
        self.v_1 = None
        self.v_2 = None
        self.nodes = None

        
    def forward(self, x):
        lambda_ = 1 + F.relu(self.lambda_param)
        
        device = self.edge_index.device        
        if self.i == None:
            self.nodes = x.shape[0]
            dummy = [i for i in range(self.nodes)]
            i_1 = torch.tensor([dummy, dummy]).to(device)
            i_2 = self.edge_index
            self.i = torch.cat([i_1, i_2], dim = 1)
            self.v_1 = torch.tensor([1 for _ in range(self.nodes)]).to(device)
            self.v_2 = torch.tensor([1 for _ in range(len(i_2[0]))]).to(device)
 
        v_1 = ((2 * lambda_ - 2) / lambda_) * self.v_1
        v_2 = (2 / lambda_) * self.v_2
        v = torch.cat([v_1, v_2])
        
        
        e_rowsum = self.special_spmm(self.i, v, torch.Size([self.nodes, self.nodes]), torch.ones(size=(self.nodes,1)).to(device))
        return self.special_spmm(self.i, v, torch.Size([self.nodes, self.nodes]), x).div(e_rowsum)

        
        # filter_matrix = torch.sparse_coo_tensor(self.i, v, torch.Size([self.nodes, self.nodes]))
        # ones_ = torch.ones(size=(self.nodes,1)).to(device)
        # D_vec_invsqrt_corr = torch.matmul(filter_matrix, ones_)
        
        # y = torch.matmul(filter_matrix, x).div(D_vec_invsqrt_corr)
        # y[torch.isinf(y)] = 0.
        
        
        # D_vec_invsqrt_corr = torch.pow(D_vec, -0.5).flatten()
        # D_vec_invsqrt_corr[torch.isinf(D_vec_invsqrt_corr)] = 0.
        # D_mat_invsqrt_corr = torch.diag(D_vec_invsqrt_corr.squeeze())
        # filter_matrix = D_mat_invsqrt_corr @ filter_matrix @ D_mat_invsqrt_corr
        
        # y = filter_matrix @ x
        
        # return y
    
    
    
    
    def conv_val(self, adj):        
        alpha = F.relu(self.lambda_param).detach().cpu().numpy()
        alpha1 = 2 * alpha / (1 + alpha)
        alpha2 = 2 / (1 + alpha)
        n_node = adj.shape[0]
        filter_matrix = alpha1 * np.eye(n_node) + alpha2 * adj
        D_vec = filter_matrix.sum(1)
        D_mean = np.mean(D_vec)
        
        return alpha1, alpha2, D_mean
            
   

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

