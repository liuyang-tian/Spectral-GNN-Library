import torch
import torch.nn as nn
import torch.nn.functional as F


class CayleyConv(nn.Module):
    def __init__(self, k, num_jacobi_iter, laplacian_matrix, input_dim, output_dim, bias):
        super(CayleyConv, self).__init__()
        self.k = k
        self.num_jacobi_iter = num_jacobi_iter
        self.laplacian_matrix = laplacian_matrix
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.h = nn.Parameter(torch.Tensor(1))
        self.real_weight = nn.Parameter(torch.Tensor((k+1) * input_dim, output_dim))
        self.imag_weight = nn.Parameter(torch.Tensor((k+1) * input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.h)
        torch.nn.init.normal_(self.real_weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.imag_weight, mean=0.0, std=1.0)
        self.bias.data.fill_(0.1)

    def forward(self, x):
        num_node = self.laplacian_matrix.shape[0]
        device = self.laplacian_matrix.device
        real_matrix = self.h * self.laplacian_matrix
        imag_matrix = torch.eye(num_node).to(device)
        
        cayley_neg = torch.complex(real_matrix, -imag_matrix)   # hL - iI
        cayley_pos = torch.complex(real_matrix, imag_matrix)   # hL + iI
        diag_cayley_pos = torch.diag(cayley_pos)
        diag_inv_cayley_pos = torch.diag_embed(torch.pow(diag_cayley_pos, -1))   # (Diag(hL + iL))^-1
        off_diag_cayley_pos =  cayley_pos - torch.diag_embed(diag_cayley_pos)   # Off(hL + iL)

        init_x = torch.complex(real=x, imag=torch.zeros_like(x))
        poly_list = [init_x]
        for i in range(self.k):
            poly_item = self.jacobi_approximate(self.num_jacobi_iter, poly_list[-1], cayley_neg, diag_inv_cayley_pos, off_diag_cayley_pos)  # N*d1
            poly_list.append(poly_item)

        poly_list = torch.cat(poly_list, dim=1)   # N * (k*d1) 
        complex_weight = torch.complex(self.real_weight, self.imag_weight)   # (k*d1) * d2

        y = poly_list @ complex_weight   # N * d2
        y = 2 * y.real   # N * d2
        y += self.bias
        return y

    def conv_val(self, eigenvals):
        h = self.h.detach().cpu().numpy() 
        complex_weight = torch.complex(self.real_weight, self.imag_weight).detach().cpu().numpy()

        order_items=[]
        order_item1 = 1.
        order_item2 = 1.
        for i in range(self.k):
            order_item1 = order_item1 * (h * eigenvals - 1j)
            order_item2 = order_item2 * (h * eigenvals + 1j)
            order_item = order_item1 / order_item2
            order_items.append(order_item)
        
        conv_vals = []
        for i in range(self.output_dim):
            input_channels = []
            for j in range(self.input_dim):
                conv_val = complex_weight[j][i]
                for k in range(self.k):
                    conv_val += complex_weight[(k+1)*self.input_dim+j][i] * order_items[k]
                conv_val = 2 * conv_val.real
                input_channels.append(conv_val)
            conv_vals.append(input_channels)

        return conv_vals
            

    def jacobi_approximate(self, num_jacobi_iter, previous_item ,cayley_neg, diag_inv_cayley_pos, off_diag_cayley_pos):
        b = diag_inv_cayley_pos @ (cayley_neg @ previous_item)
        y = b
        for i in range(num_jacobi_iter):
            y = - diag_inv_cayley_pos @ (off_diag_cayley_pos @ y) + b
        
        return y 

