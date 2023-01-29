import torch.nn as nn
import torch


class LanczosConv(nn.Module):
    def __init__(self, long_diff_len, short_diff_len, mlp_dim, hidden_dim):
        super(LanczosConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(long_diff_len, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, long_diff_len)
        )

        self.linear = nn.Linear(
            ((long_diff_len + short_diff_len) * hidden_dim), hidden_dim)

    def forward(self, laplacian_matrix, eigen_vecs, x, short_diffusion_list, eigen_vals_long_pow):
        '''
            Args:
                laplacian_matrix: N * N
                eigen_vecs: N * K
                x: N * d
                eigen_vals_long_pow: K * E

            Returns:
                y: N *d
        '''

        # long diffusion
        filter = self.mlp(eigen_vals_long_pow)   # k*long_diff_len
        spectral_trans = eigen_vecs.T @ x   # k*d
        long_x = filter.unsqueeze(
            2) * spectral_trans.unsqueeze(1)   # k*long_diff_len*d
        k = filter.shape[0]
        y = eigen_vecs @ long_x.view(k, -1)   # N*long_diff_len*d

        # N*((short_diff_len+long_diff_len)*d)

        if len(short_diffusion_list) != 0:
            # short diffusion
            short_x = []
            tmp_x = x
            max_short_list = max(short_diffusion_list)
            for S_i in range(max_short_list+1):
                if S_i != 0:
                    tmp_x = laplacian_matrix @ tmp_x
                if S_i in short_diffusion_list:
                    short_x.append(tmp_x)
            short_x = torch.cat(short_x, dim=1)   # N*(M*d)
            y = torch.cat((short_x, y), dim=1)
            
        y = self.linear(y)

        return y
    
    def conv_val(self, eigen_vals_long_pow):
        filter = (self.mlp(eigen_vals_long_pow)).detach().cpu().numpy()   # k*long_diff_len
        weight_matrix = (self.linear.weight.data).detach().cpu().numpy()
        return filter, weight_matrix



