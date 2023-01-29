import torch
import torch.nn as nn
import numpy as np


class ARMAConv(nn.Module):
    def __init__(self, stack_num, stack_layer_num, laplacian_matrix, activation, bias, dprate, input_dim, hidden_dim):
        super(ARMAConv, self).__init__()
        self.dprate = dprate 
        self.activation = activation
        self.stack_num = stack_num
        self.stack_layer_num = stack_layer_num
        self.laplacian_matrix = laplacian_matrix

        self.W_0 = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for i in range(stack_num)])
        self.W = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(stack_num)])
        self.V = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for i in range(stack_num)])

        if bias:
            self.bias = nn.Parameter(torch.Tensor(stack_num, stack_layer_num, 1, hidden_dim))
            self.reset_parameters()

        else:
            self.register_parameter('bias', None)
        
    def reset_parameters(self):
        self.bias.data.fill_(0)


    def forward(self, x):
        init_x = x

        y = 0.
        for stack in range(self.stack_num):
            stack_out_feat = self.stack_out(init_x, stack)
            y += stack_out_feat
        
        y = y / self.stack_num

        return y
    
    def stack_out(self, init_x, stack):
        stack_out_feat = init_x
        for step in range(self.stack_layer_num):
            feat = self.laplacian_matrix @ stack_out_feat
            if step == 0:
                layer_conv_feat = self.W_0[stack](feat)
            else:
                layer_conv_feat = self.W[stack](feat)
            
            stack_out_feat = layer_conv_feat
            # stack_out_feat += nn.functional.dropout(self.V[stack](init_x), p=self.dprate, training=True)
            # stack_out_feat += self.V[stack](nn.functional.dropout(init_x, p=self.dprate, training=True))
            stack_out_feat += self.V[stack](init_x)

            if self.bias is not None:
                stack_out_feat += self.bias[stack][step]
                
            if self.activation is not None:
                stack_out_feat = self.activation(stack_out_feat)

        return stack_out_feat
    
    def conv_val(self, in_x, eigenvecs):
        eigenvecs = torch.Tensor(eigenvecs).to(in_x.device)
        stack_conv_vals = []
        for i in range(self.stack_num):
            out_x = self.stack_out(in_x, i)
            spec_out = (eigenvecs.T @ out_x).sum(1)
            spec_in = (eigenvecs.T @ in_x).sum(1)
            i_conv_vals = in_x.shape[1] / out_x.shape[1] * ( spec_out / spec_in )
            stack_conv_vals.append(i_conv_vals)
        return stack_conv_vals



