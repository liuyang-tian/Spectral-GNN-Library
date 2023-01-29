import numpy as np
import torch
import torch.nn as nn


class JacobiConv(nn.Module):
    def __init__(self, a, b, base_alpha, k, channels, laplacian_matrix):
        super(JacobiConv, self).__init__()
        self.k = k
        self.a = a
        self.b = b
        self.laplacian_matrix = laplacian_matrix
        self.base_alpha = base_alpha
        self.conv_params = nn.ParameterList([nn.Parameter(torch.tensor(float(min(1 / base_alpha, 1.0)))) for i in range(k+1)])
        self.comb_params = nn.Parameter(torch.ones((k+1, channels)))    # k+1 * d

    def forward(self, x):
        alpha_coefs = [self.base_alpha * torch.tanh(i) for i in self.conv_params]
        jacobi_polys=[]
        for i in range(self.k+1):
            item = self.jacobi_poly(i, x, self.laplacian_matrix, jacobi_polys, self.a, self.b, alpha_coefs)
            jacobi_polys.append(item)   # N * d
        
        jacobi_polys = torch.stack(jacobi_polys, dim=0)    # k+1 * N * d
        
        y = self.comb_params.unsqueeze(dim=1) * jacobi_polys    # k+1 * N * d
        y = torch.sum(y, dim=0)    # N * d

        return y

    def jacobi_poly(self, index, x, multiplicator, results, a, b, alpha_coefs, l=-1.0, r=1.0):
        if index==0:
            return x
        elif index==1:
            coef1 = alpha_coefs[0] * (((a - b) / 2) - (((a + b + 2) / 2) * ((r + l) / (r - l))))
            coef2 = alpha_coefs[0] * ((a + b + 2) / (r - l))
            if len(multiplicator.shape) > 1:
                r = coef1 * x + coef2 * multiplicator @ x
            else:
                r = coef1 * x + coef2 * multiplicator * x
            return r
        else:
            coef_divisor = 2 * index * (index + a + b) * (2 * index + a + b -2)
            coef_1 = (2 * index + a + b) * (2 * index + a + b - 1) * (2 * index + a + b -2)
            coef_2 = (2 * index + a + b - 1) * (a**2 - b**2)
            coef_3 = 2 * (index + a - 1) * (index + b - 1) * (2 * index + a + b)
            coef_1 = alpha_coefs[index-1] * (coef_1 / coef_divisor)
            coef_2 = alpha_coefs[index-1] * (coef_2 / coef_divisor)
            coef_3 = alpha_coefs[index-1] * alpha_coefs[index-2] * (coef_3 / coef_divisor)
            coef1 = coef_1 * (2 / (r - l))
            coef2 = coef_1 * ((r + l) / (r - l)) + coef_2
            if len(multiplicator.shape) > 1:
                r = coef1 * multiplicator @ results[-1] + coef2 * results[-1] - coef_3 * results[-2]
            else:
                r = coef1 * multiplicator * results[-1] + coef2 * results[-1] - coef_3 * results[-2]
            return r
    
    def conv_val(self, eigenvals):
        alpha_coefs = [(self.base_alpha * torch.tanh(i)).detach().cpu().numpy() for i in self.conv_params]
        jacobi_polys=[]
        iden = np.ones_like(eigenvals)
        for i in range(self.k+1):
            item = self.jacobi_poly(i, iden, eigenvals, jacobi_polys, self.a, self.b, alpha_coefs)
            jacobi_polys.append(item) 
        
        comb_params = (self.comb_params).detach().cpu().numpy()
        conv_vals = []
        for i in range(comb_params.shape[1]):
            conv_val = 0.
            for k in range(self.k+1):
                conv_val += comb_params[k,i] * jacobi_polys[k]

            conv_vals.append(conv_val)
        return conv_vals
        





            
            


