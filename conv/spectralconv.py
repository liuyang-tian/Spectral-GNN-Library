import torch
import torch.nn as nn


class SpecConv(nn.Module):
    def __init__(self, dim1, dim2, K):
        super(SpecConv, self).__init__()
        self.filter = nn.Parameter(torch.Tensor(K, dim1, dim2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.filter)

    def forward(self, x, v_k):
        """
        Args:
            x: [N, d1]
            v_k: [N, k]

        Returns: 
            y: [N, d2]
        """

        spectral_trans = v_k.T @ x    # [k, d1]
        # [k, d1, d2] . [k, d1, 1]
        filter = self.filter * spectral_trans.unsqueeze(2)
        filter = torch.sum(filter, axis=1)    # [k, d2]
        y = v_k @ filter    # [N, k] * [k, d2] = [N, d2]

        return y
    
    def conv_val(self):
        return self.filter
