import torch
import torch.nn as nn
import numpy as np


class ChebyConv(nn.Module):
    def __init__(self, dim1, dim2, k):
        super(ChebyConv, self).__init__()
        self.k = k
        self.linear = nn.Linear(dim1 * (k+1), dim2)
        self.dim1 = dim1
        self.dim2 = dim2
        

    def forward(self, scaled_laplacian, x):

        Cheby_poly = self.get_cheby_poly(x, scaled_laplacian)
        x = torch.cat(Cheby_poly, dim=1)

        y = self.linear(x)
        return y

    def get_cheby_poly(self, p0, multiplicator):
        # cheby polynomial for Laplacian
        Cheby_poly = []
        for i in range(0, self.k+1):
            if i == 0:
                Cheby_poly.append(p0)
            elif i == 1:
                if len(multiplicator.shape) > 1:   # laplacian matrix
                    Cheby_poly.append(multiplicator @ p0)
                else:
                    Cheby_poly.append(multiplicator * p0)
            else:
                if len(multiplicator.shape) > 1:   # laplacian matrix
                    x_i = 2 * multiplicator @ Cheby_poly[i-1] - Cheby_poly[i-2]
                else:
                    x_i = 2 * multiplicator * Cheby_poly[i-1] - Cheby_poly[i-2]
                Cheby_poly.append(x_i)
        
        return Cheby_poly
    

    def conv_val(self, eigenvals):
        output_channel = []
        iden = np.ones_like(eigenvals)
        multiplicator = eigenvals / max(eigenvals) - 1
        cheby_poly = self.get_cheby_poly(iden, multiplicator)

        conv_param = self.linear.weight.data.cpu().numpy()   # dim1 * (k+1), dim2
        for i in range(self.dim2):
            input_channel = []
            for j in range(self.dim1):
                conv_val = 0.
                for k in range(self.k+1):
                    conv_val += cheby_poly[k] * conv_param[i][k*self.dim1+j]
                input_channel.append(conv_val)
            output_channel.append(input_channel)
        return output_channel


