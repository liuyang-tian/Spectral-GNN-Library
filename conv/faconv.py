import torch
import torch.nn as nn
from dgl import function as fn
import torch.nn.functional as F


class FALayer(nn.Module):
    def __init__(self, g, in_dim, dprate):
        super(FALayer, self).__init__()
        self.g = g
        self.dprate = dprate
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)


    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        e = F.dropout(e, p=self.dprate, training=self.training)
        return {'e': e, 'm': g}


    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']