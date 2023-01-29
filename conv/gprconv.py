import numpy as np
import torch
import torch.nn as nn


class GPRConv(nn.Module):
    def __init__(self, init_method, alpha, k, Gamma, laplacian_matrix):
        super(GPRConv, self).__init__()
        self.k = k
        self.laplacian_matrix = laplacian_matrix

        assert init_method in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if init_method == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            gamma = 0.0*np.ones(k+1)
            gamma[alpha] = 1.0
        elif init_method == 'PPR':
            # PPR-like
            gamma = alpha*(1-alpha)**np.arange(k+1)
            gamma[-1] = (1-alpha)**k
        elif init_method == 'NPPR':
            # Negative PPR
            gamma = (alpha)**np.arange(k+1)
            gamma = gamma/np.sum(np.abs(gamma))
        elif init_method == 'Random':
            # Random
            bound = np.sqrt(3/(k+1))
            gamma = np.random.uniform(-bound, bound, k+1)
            gamma = gamma/np.sum(np.abs(gamma))
        elif init_method == 'WS':
            # Specify Gamma
            gamma = Gamma

        self.gamma = nn.Parameter(torch.Tensor(gamma))


    def forward(self, x):
        y = 0.

        for step in range(self.k+1):
            if step == 0:
                layer_conv_output = x
            else:
                layer_conv_output = self.laplacian_matrix @ layer_conv_output
            
            y += self.gamma[step] * layer_conv_output

        return y
    
    def conv_val(self, eigenvals):
        gamma = self.gamma.detach().cpu().numpy()
        conv_base = np.ones_like(eigenvals)
        conv_val = gamma[0] * conv_base
        for k in range(self.k):
            conv_base = conv_base * eigenvals
            conv_val += gamma[k+1] * conv_base
        return conv_val



