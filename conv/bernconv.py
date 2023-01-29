import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb


class BernConv(nn.Module):
    def __init__(self, k, laplacian_matrix):
        super(BernConv, self).__init__()
        self.k = k
        self.laplacian_matrix = laplacian_matrix
        self.filter_param = nn.Parameter(torch.Tensor(k+1))
        self.reset_parameters()
    
    def reset_parameters(self):
        self.filter_param.data.fill_(1)

    def forward(self, x):
        filter_param = F.relu(self.filter_param)

        # 2I - L
        num_node = self.laplacian_matrix.shape[0]
        device = self.laplacian_matrix.device
        poly_item = 2 * torch.eye(num_node).to(device) - self.laplacian_matrix

        y = self.get_bern_poly(poly_item, self.laplacian_matrix, x, filter_param)
        # first_poly_list = [x]
        # i_pow_poly = x
        # for i in range(self.k):
        #     i_pow_poly = poly_item @ i_pow_poly
        #     first_poly_list.append(i_pow_poly)
        
        # y = 0.
        # for i in range(self.k+1):
        #     filter_poly = first_poly_list[self.k - i]
        #     for j in range(i):
        #         filter_poly = self.laplacian_matrix @ filter_poly

        #     y +=  filter_param[i] * comb(self.k, i) / (2**i) * filter_poly

        return y

    def get_bern_poly(self, poly_item1, poly_item2, x, filter_param):
        first_poly_list = [x]
        i_pow_poly = x
        for i in range(self.k):
            if len(poly_item1.shape) > 1:
                i_pow_poly = poly_item1 @ i_pow_poly
            else:
                i_pow_poly = poly_item1 * i_pow_poly
            first_poly_list.append(i_pow_poly)
        
        y = 0.
        for i in range(self.k+1):
            filter_poly = first_poly_list[self.k - i]
            for j in range(i):
                if len(poly_item2.shape) > 1:
                    filter_poly = poly_item2 @ filter_poly
                else:
                    filter_poly = poly_item2 * filter_poly

            y +=  filter_param[i] * comb(self.k, i) / (2**self.k) * filter_poly

        return y


    def conv_val(self, eigenvals):       
        filter_param = F.relu(self.filter_param).detach().cpu().numpy()
        poly_item1 = 2 - eigenvals
        conv_vals = self.get_bern_poly(poly_item1, eigenvals, 1., filter_param)
        return conv_vals



    
