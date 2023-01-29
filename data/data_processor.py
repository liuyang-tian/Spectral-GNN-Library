import torch
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
from scipy import sparse as sp
import os
from numpy.linalg import eigh


class DataProcessor():
    def __init__(self, data, train_rate, val_rate, data_process):

        self.name = data.name
        self.num_features = data.num_features
        self.num_classes = data.num_classes
        self.num_nodes = data.num_nodes
        self.edge_index = data.edge_index

        self.x = data.x
        self.y = data.y

        self.train_mask, self.val_mask, self.test_mask = self.data_split(train_rate, val_rate)
        
        self.eigenvalues = None
        self.eigenvectors = None

        laplacian_type = data_process.laplacian_type if "laplacian_type" in data_process else "L2"
        add_self_loop = data_process.add_self_loop if "add_self_loop" in data_process else False
        epsilon = -0.5 if "epsilon" not in data_process or laplacian_type=="L0" or laplacian_type=="L1" else data_process.epsilon
        need_EVD = data_process.need_EVD if "need_EVD" in data_process else False
        eigen_sorted = data_process.eigen_sorted if "eigen_sorted" in data_process else False

        adj = to_scipy_sparse_matrix(self.edge_index)
        laplacian_matrix = self.get_Laplacian(adj, laplacian_type, add_self_loop, epsilon)
        laplacian_matrix = laplacian_matrix.todense()
        if need_EVD:
            pre_filename = self.get_pre_filename(laplacian_type, epsilon, add_self_loop)
            eigenvalues, eigenvectors = self.get_eigh(laplacian_matrix, eigen_sorted, self.name, "eigh", pre_filename)
            self.eigenvalues=torch.Tensor(eigenvalues)
            self.eigenvectors=torch.Tensor(eigenvectors)
            
        self.laplacian_matrix = torch.Tensor(laplacian_matrix)
    
    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        self.laplacian_matrix = self.laplacian_matrix.to(device)
        
        if not self.eigenvalues is None and not self.eigenvectors is None:
            self.eigenvalues = self.eigenvalues.to(device)
            self.eigenvectors = self.eigenvectors.to(device)

        return self
    
    @classmethod
    def get_pre_filename(cls, laplacian_type, epsilon, add_self_loop):
        self_loop = "_self_loop" if add_self_loop else ""
        return laplacian_type + "_" + str(epsilon) + self_loop
    
    
    @classmethod
    def get_Laplacian(cls, adj, laplacian_type="L2", add_self_loop=False, epsilon=-0.5):
        """
        Compute Graph Laplacian

        Args:
            laplacian_type:
            -L0: directly use A
            -L1: use combinatorial graph Laplacian, L = D - A
            -L2: use symmetric graph Laplacian, L = I - D^{-1/2} ( A  or  I + A ) D^{-1/2}
            -L3: random-walk normalized Laplacian matrix, L = D^(-1) @ L

        Returns:
            L: N X N, graph Laplacian matrix
        """
        n_node = adj.shape[0]
        identity_mat = sp.eye(n_node) if sp.issparse(adj) else np.eye(n_node)
        if add_self_loop:
            adj = identity_mat + adj

        D_vec = np.array(adj.sum(1))
        D_mat = sp.diags(D_vec.flatten()) if sp.issparse(adj) else np.diag(D_vec.flatten())

        D_vec_invsqrt_corr = np.power(D_vec, epsilon).flatten()
        D_vec_invsqrt_corr[np.isinf(D_vec_invsqrt_corr)] = 0.
        D_mat_invsqrt_corr = sp.diags(D_vec_invsqrt_corr.squeeze()) if sp.issparse(adj) else np.diag(D_vec_invsqrt_corr.squeeze())
        
        if laplacian_type == 'L0':
            L = adj
        elif laplacian_type == 'L1':
            # L = D - A
            L = D_mat - adj
        elif laplacian_type == 'L2':
            # L = I - D^{-1/2} A D^{-1/2}
            L = identity_mat - D_mat_invsqrt_corr @ adj @ D_mat_invsqrt_corr
        elif laplacian_type == 'L3':
            # L = D^{epsilon} ( A or I + A ) D^{epsilon}
            L = D_mat_invsqrt_corr @ adj @ D_mat_invsqrt_corr
        elif laplacian_type == 'L4':
            # L = D^(-1) @ L
            L = D_mat_invsqrt_corr @ (D_mat - adj)
        elif laplacian_type == 'L5':
            # L = D^(-1) @ A
            L = D_mat_invsqrt_corr @ adj
        else:
            raise ValueError('Unsupported Graph Laplacian!')

        return L

    @classmethod
    def get_eigh(cls, laplacian_matrix, eigen_sorted, data_name, file_dir, pre_name, save=True):

        dir='../datasets/'+data_name+'/' + file_dir
        if not os.path.isdir(dir):
            os.makedirs(dir)

        val_file_name = pre_name + '_eigenvalues.npy' 
        vec_file_name = pre_name + '_eigenvectors.npy' 
        
        eigvals_path=os.path.join(dir, val_file_name)
        eigvecs_path=os.path.join(dir, vec_file_name)

        if os.path.exists(eigvals_path) and os.path.exists(eigvecs_path):
            eigenvalues = np.load(eigvals_path)
            eigenvectors = np.load(eigvecs_path)
        else:
            if sp.issparse(laplacian_matrix):
                laplacian_matrix = laplacian_matrix.todense()
            eigenvalues, eigenvectors = eigh(laplacian_matrix)
            
            if save:
                np.save(eigvals_path, eigenvalues)
                np.save(eigvecs_path, eigenvectors)
        
        if eigen_sorted:
            idx = np.argsort(eigenvalues, kind='mergesort')
            eigenvalues = eigenvalues[idx[:]]
            eigenvectors = eigenvectors[:, idx[:]]

        return eigenvalues, eigenvectors

    def index_to_mask(self, index):
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        mask[index] = 1
        return mask

    def data_split(self, train_rate, val_rate):
        percls_trn=int(round(train_rate*len(self.y)/self.num_classes))
        val_lb=int(round(val_rate*len(self.y)))
        
        index = [i for i in range(0, len(self.y))]
        train_idx = []
        for c in range(self.num_classes):
            class_idx = np.where(self.y.cpu() == c)[0]
            if len(class_idx) < percls_trn:
                train_idx.extend(class_idx)
            else:
                train_idx.extend(np.random.choice(
                    class_idx, percls_trn, replace=False))
        
        rest_idx = [i for i in index if i not in train_idx]
        val_idx = np.random.choice(rest_idx, val_lb, replace=False)
        test_idx = [i for i in rest_idx if i not in val_idx]

        train_mask = self.index_to_mask(train_idx)
        val_mask = self.index_to_mask(val_idx)
        test_mask = self.index_to_mask(test_idx)

        return train_mask, val_mask, test_mask
    

    