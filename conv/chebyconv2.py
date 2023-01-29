import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class ChebyConv2(nn.Module):
    def __init__(self, k):
        super(ChebyConv2, self).__init__()
        self.k = k
        self.filter_param = nn.Parameter(torch.Tensor(k+1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.filter_param)

    def forward(self, chebynodes_vals, scaled_laplacian, x):
        """
        Args:
            chebynodes_vals: k+1 * k+1

        """

        filter_param = F.relu(self.filter_param)
        filter_param = chebynodes_vals @ filter_param    # (k+1)*1
        filter_param[0] = filter_param[0] / 2

        # cheby polynomial for Laplacian
        # cheby_poly = []
        # for i in range(0, self.k+1):
        #     if i == 0:
        #         cheby_poly.append(x)
        #     elif i == 1:
        #         cheby_poly.append(scaled_laplacian @ x)
        #     else:
        #         x_i = 2 * scaled_laplacian @ cheby_poly[i-1] - cheby_poly[i-2]
        #         cheby_poly.append(x_i)

        cheby_poly = self.get_cheby_poly(x, scaled_laplacian)
        cheby_poly = torch.stack(cheby_poly)    # (k+1)*N*d

        filter_param = filter_param.unsqueeze(dim=2) * cheby_poly    # (k+1)*N*d
        y = filter_param.sum(dim=0)    # N*d
        y = 2/(self.k+1) * y

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


    def conv_val(self, chebynodes_vals, eigenvals):
        filter_param = F.relu(self.filter_param).detach().cpu().numpy()
        chebynodes_vals = chebynodes_vals.cpu().numpy()
        filter_param = chebynodes_vals @ filter_param    # (k+1)*1
        filter_param[0] = filter_param[0] / 2

        iden = np.ones_like(eigenvals)
        multiplicator = eigenvals / max(eigenvals) - 1
        cheby_poly = self.get_cheby_poly(iden, multiplicator)
        cheby_poly = np.vstack(cheby_poly)

        conv_vals = 2 / (self.k+1) * (filter_param.T @ cheby_poly)

        return conv_vals





