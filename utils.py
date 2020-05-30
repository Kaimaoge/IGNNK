import os
import zipfile
import numpy as np
import scipy.sparse as sp

"""
Load datasets
"""

def load_metr_la_rdata():
    if (not os.path.isfile("data/adj_mat.npy")
            or not os.path.isfile("data/node_values.npy")):
        with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    A = np.load("data/adj_mat.npy")
    X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    return A, X

"""
Dingyi please add reading other datasets here
"""

"""
Dynamically construct the adjacent matrix
"""

def get_Laplace(A):
    """
    Returns the laplacian adjacency matrix. This is for C_GCN
    """
    if A[0, 0] == 1:
        A = A - np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()
