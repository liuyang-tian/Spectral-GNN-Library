import torch.nn as nn
import torch


class CorrelationFreeConv(nn.Module):
    def __init__(self, k, hidden_dim):
        super(CorrelationFreeConv, self).__init__()
        self.filter_parm = nn.Parameter(torch.Tensor(k+1, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.filter_parm)

    def forward(self, filter_basis, x):
        filter_x = self.filter_parm.unsqueeze(1) * x.unsqueeze(0)    # k*1*d * 1*N*d 
        y = filter_basis @ filter_x
        y = y.sum(dim=0)

        return y

    def conv_val(self):
        filter_param = (self.filter_parm).detach().cpu().numpy()
        return filter_param
