import torch
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from utils import setup_features

class PreProcess(object):
    def __init__(self, edges, args):
        self.edges = edges
        self.args = args
        # Preprocess on the CPU for scalability
        self.device = torch.device("cpu")

    def preprocess(self):
        self.positive_edges, self.negative_edges, self.test_positive_edges, self.test_negative_edges, self.y = self.setup_dataset()

        self.Z = torch.zeros_like(self.X)

        # Symmetric normalization of adjacency matrices
        self.A_P = self.symmetric_normalize(self.edges['positive_adjacency_matrix'])
        self.A_N = self.symmetric_normalize(self.edges['negative_adjacency_matrix'])

        # Initialize hidden representations
        X_P = self.X
        X_N = self.X

        for k in range(self.args.num_layers):
            # Apply the LightSGCN propagation
            X_P_new = torch.mm(self.A_P, X_P) + torch.mm(self.A_N, X_N)
            X_N_new = torch.mm(self.A_P, X_N) + torch.mm(self.A_N, X_P)
            
            # Normalize the hidden representations using L2 normalization
            X_P = self.l2_normalize(X_P_new)
            X_N = self.l2_normalize(X_N_new)

            # Combine the final representations from all layers
            self.Z += (1 - self.args.alpha_0) / self.args.num_layers * (X_P - X_N)

        # Combine the final representations from all layers
        self.Z += self.args.alpha_0 * self.X

        # Compute the Prob_vi for each node
        prob_vi = self.compute_prob_vi()

        return self.Z, prob_vi, self.positive_edges, self.negative_edges, self.test_positive_edges, self.test_negative_edges, self.y

    def symmetric_normalize(self, adjacency_matrix):
        """
        Symmetrically normalize the adjacency matrix.
        Ã = D^(-1/2) * A * D^(-1/2)
        """
        adjacency_matrix = self.convert_to_sparse_tensor(adjacency_matrix)
        
        degree_matrix = torch.sum(adjacency_matrix.to_dense(), dim=1)  # Calculate the degree matrix
        degree_inv_sqrt = torch.pow(degree_matrix, -0.5)  # D^(-1/2)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0  # Handle division by 0
        
        # Normalize the adjacency matrix Ã = D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        normalized_adj = torch.mm(D_inv_sqrt, torch.mm(adjacency_matrix.to_dense(), D_inv_sqrt))
        
        return normalized_adj

    def compute_prob_vi(self):
        """
        Compute Prob_vi for each node using the formula:
        Prob_vi = d^P_vi / (d^P_vi + d^N_vi)
        """
        # Convert sparse adjacency matrices to dense
        A_P_dense = self.A_P.to_dense()
        A_N_dense = self.A_N.to_dense()

        # Compute the degree for each node in positive and negative graphs
        d_P = torch.sum(A_P_dense, dim=1)  # Degree in positive graph
        d_N = torch.sum(A_N_dense, dim=1)  # Degree in negative graph

        # Ensure no division by zero: add a small epsilon to the denominator
        prob_vi = d_P / (d_P + d_N + 1e-10)

        return prob_vi

    def convert_to_sparse_tensor(self, adjacency_matrix):
        """
        Convert a SciPy sparse matrix (csr_matrix or coo_matrix) to a PyTorch sparse tensor.
        """
        if isinstance(adjacency_matrix, scipy.sparse.csr_matrix):
            adjacency_matrix = adjacency_matrix.tocoo()  # Convert to COO format
        elif not isinstance(adjacency_matrix, scipy.sparse.coo_matrix):
            raise ValueError("Input should be a SciPy csr_matrix or coo_matrix")

        # Use numpy.vstack to combine row and col indices into a single numpy array before converting to tensor
        indices = torch.LongTensor(np.vstack((adjacency_matrix.row, adjacency_matrix.col)))
        values = torch.FloatTensor(adjacency_matrix.data)
        shape = torch.Size(adjacency_matrix.shape)

        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
        return sparse_tensor


    def l2_normalize(self, X):
        """
        Normalize the matrix X row-wise using the L2 norm.
        """
        norm = torch.norm(X, p=2, dim=1, keepdim=True)
        norm[norm == 0] = 1e-10
        return X / norm

    def setup_dataset(self):
        """
        Creating train and test split.
        """
        self.positive_edges, self.test_positive_edges = train_test_split(self.edges["positive_edges"],
                                                                        test_size=self.args.test_size)

        self.negative_edges, self.test_negative_edges = train_test_split(self.edges["negative_edges"],
                                                                        test_size=self.args.test_size)
        
        self.ecount = len(self.positive_edges + self.negative_edges)

        self.X = setup_features(self.args,
                                self.positive_edges,
                                self.negative_edges,
                                self.edges["ncount"])

        self.positive_edges = torch.from_numpy(np.array(self.positive_edges,
                                                        dtype=np.int64).T).type(torch.long).to(self.device)

        self.negative_edges = torch.from_numpy(np.array(self.negative_edges,
                                                        dtype=np.int64).T).type(torch.long).to(self.device)

        self.y = np.array([0 if i < int(self.ecount/2) else 1 for i in range(self.ecount)]+[2]*(self.ecount*2))
        self.y = torch.from_numpy(self.y).type(torch.LongTensor).to(self.device)
        self.X = torch.from_numpy(self.X).float().to(self.device)

        return self.positive_edges, self.negative_edges, self.test_positive_edges, self.test_negative_edges, self.y