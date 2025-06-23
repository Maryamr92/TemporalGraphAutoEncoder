import torch
import numpy as np
import tensorly as tl
import tensorly.decomposition as decom
from tensorly.cp_tensor import cp_to_tensor
# from tensorly.contrib.sparse import tensor as sparse_tensor
from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac
from scipy.sparse import coo_matrix
import sparse                      # Sparse array library, NOT PyTorch
import tensorly.contrib.sparse as stl
# from tensorly.contrib.sparse.decomposition import parafac


# tl.set_backend('numpy')  # Can be changed to 'pytorch' if needed
tl.set_backend('pytorch')


### === DENSE PARAFAC === ###
def parafac_decomposition_dense(adj_tensor, rank, n_iter_max=100, device=None):
    """
    Apply dense PARAFAC decomposition to a 3D tensor.

    Args:
        tensor (np.ndarray): Input dense tensor.
        rank (int): Number of components.
        n_iter_max (int): Max number of iterations.

    Returns:
        A, B, C (torch.Tensor): Factor matrices.
    """
    # Ensure tensor is on the desired device
    adj_tensor = adj_tensor.to(device)

    adj_tensor = np.squeeze(adj_tensor)
    adj_tensor = tl.tensor(adj_tensor)

    weights, factors = decom.parafac(
        adj_tensor,
        rank=rank,
        normalize_factors=True,
        n_iter_max=n_iter_max,
        tol=1e-6,
        init='svd'
    )
    # print(f'weight dense, {weights}')

    # Sort by descending weights
    sorted_idx = torch.argsort(-weights)
    sorted_factors = [f[:, sorted_idx] for f in factors]

    sorted_weights =  weights[sorted_idx]

    # print(f'sorted_weights dense, {sorted_weights}')
    A, B, C = sorted_factors

    # A, B, C = [torch.from_numpy(f) for f in sorted_factors]

    # print(f'A Dense matrix,  {A}')
    # print(f'B Dense matrix,  {B}')
    # print(f'C Dense matrix,  {C}')

    reconstructed_tensor = cp_to_tensor((weights, factors))
    error = tl.norm(adj_tensor - reconstructed_tensor)

    # print("Reconstruction Error dense (Frobenius norm):", error)

    return A, B, C


def parafac_decomposition_list_dense(tensor_list, rank, device=None):
    """
    Apply dense PARAFAC to a list of tensors.

    Args:
        tensor_list (list of np.ndarray): List of 3D tensors.
        rank (int): Rank of decomposition.

    Returns:
        list of [A, B, C] factors (torch.Tensors)
    """

    factors_list =[]

    for tensor in tensor_list:

        A, B, C = parafac_decomposition_dense(tensor, rank, n_iter_max=100, device=device)

        factors_list.append([A.to(device), B.to(device), C.to(device)])

    return factors_list


### === SPARSE PARAFAC === ###
def parafac_decomposition_sparse(adj_tensor, rank, n_iter_max=100, device=None):
    """
    Apply sparse PARAFAC to a tensor represented in COO format.

    Args:
        indices (np.ndarray): Shape (ndim, nnz)
        values (np.ndarray): Shape (nnz,)
        shape (tuple): Tensor shape
        rank (int): Rank of decomposition
        n_iter_max (int): Max number of iterations

    Returns:
        A, B, C (torch.Tensor): Factor matrices.
    """
    # tensorly.contrib.sparse and pydata/sparse are CPU-only.

    # Ensure the tensor is on CPU before converting to numpy
    adj_tensor_cpu = adj_tensor.cpu()
    dense_tensor = adj_tensor_cpu.numpy()

    # Get coordinates of nonzero elements
    coords = np.array(np.nonzero(dense_tensor))  # shape (ndim, nnz)

    # Get data (values) at those coords
    data = dense_tensor[tuple(coords)]

    # Create sparse COO tensor
    T_sparse = sparse.COO(coords, data, shape=adj_tensor.shape)

    tensorly_tensor = stl.tensor(T_sparse, dtype='float')

    weights, factors = sparse_parafac(
        tensorly_tensor,
        rank=rank,
        normalize_factors=True,
        n_iter_max=n_iter_max,
        init='random'
    )



    # Make sure weights is a NumPy array (dense)
    if isinstance(weights, sparse.COO):
        weights = weights.todense()

    weights = np.array(weights)  # In case it's still not NumPy

    # print(f'weight sparse, {weights}')

    # Now sort
    sorted_idx = np.argsort(-weights)
    sorted_factors = [f[:, sorted_idx] for f in factors]

    sorted_weights = weights[sorted_idx]

    # print(f'sorted_weights sparse, {sorted_weights}')

    # Step 3: Convert to PyTorch tensors
    A, B, C = [torch.tensor(f, dtype=torch.float32) for f in sorted_factors]

    # print(f'A sparse matrix,  {A}')
    # print(f'B sparse matrix,  {B}')
    # print(f'C sparse matrix,  {C}')

    reconstructed_tensor = cp_to_tensor((weights, factors))
    tensorly_tensor = tl.tensor(dense_tensor, dtype='float')

    error = tl.norm(tensorly_tensor - reconstructed_tensor)

    # print("Reconstruction Error sparse (Frobenius norm):", error)



    return A, B, C


def parafac_decomposition_list_sparse(tensor_list, rank, device=None):
    """
    Apply sparse PARAFAC decomposition to a list of sparse tensors.

    Args:
        sparse_adj_list (list of dict): Each dict has 'indices', 'values', 'shape'.
        rank (int): Rank of decomposition.

    Returns:
        List of [A, B, C] factor tensors for each input tensor.
    """
    factors_list = []
    for tensor in tensor_list:

        A, B, C = parafac_decomposition_sparse(tensor, rank, n_iter_max=100, device=device)
        factors_list.append([A.to(device), B.to(device), C.to(device)])

    return factors_list



import torch

def sparse_parafac_gpu(tensor, rank, n_iter=50, device='cuda'):
    """
    Perform sparse CP decomposition (PARAFAC) using ALS on GPU.
    
    Args:
        tensor (torch.sparse.Tensor): 3D sparse tensor (shape: I x J x K)
        rank (int): CP rank
        n_iter (int): Number of ALS iterations
        device (str): 'cuda' or 'cpu'

    Returns:
        A, B, C: Low-rank factor matrices (I x R, J x R, K x R)
    """
    assert tensor.is_sparse, "Input must be a sparse tensor"
    assert tensor.dim() == 3, "Only 3D tensors supported"

    I, J, K = tensor.shape
    indices = tensor._indices()  # shape: (3, nnz)
    values = tensor._values()    # shape: (nnz,)

    # Initialize factor matrices on GPU
    A = torch.rand(I, rank, device=device, requires_grad=False)
    B = torch.rand(J, rank, device=device, requires_grad=False)
    C = torch.rand(K, rank, device=device, requires_grad=False)

    # Precompute index mappings
    i_idx, j_idx, k_idx = indices

    for _ in range(n_iter):
        # === Update A ===
        for i in range(I):
            mask = (i_idx == i)
            if mask.sum() == 0:
                continue
            bj = B[j_idx[mask]]
            ck = C[k_idx[mask]]
            V = bj * ck  # shape: (nnz_i, rank)
            A[i] = torch.linalg.lstsq(V, values[mask].unsqueeze(1)).solution.squeeze()

        # === Update B ===
        for j in range(J):
            mask = (j_idx == j)
            if mask.sum() == 0:
                continue
            ai = A[i_idx[mask]]
            ck = C[k_idx[mask]]
            V = ai * ck
            B[j] = torch.linalg.lstsq(V, values[mask].unsqueeze(1)).solution.squeeze()

        # === Update C ===
        for k in range(K):
            mask = (k_idx == k)
            if mask.sum() == 0:
                continue
            ai = A[i_idx[mask]]
            bj = B[j_idx[mask]]
            V = ai * bj
            C[k] = torch.linalg.lstsq(V, values[mask].unsqueeze(1)).solution.squeeze()

    return A, B, C



def parafac_decomposition_list_sparse_gpu(tensor_list, rank, n_iter=50, device='cuda'):
    """
    Apply GPU-based sparse PARAFAC decomposition (ALS) to a list of sparse 3D tensors.

    Args:
        tensor_list (list of torch.sparse.Tensor): List of sparse 3D tensors (must be COO format).
        rank (int): Rank of CP decomposition.
        n_iter (int): Number of ALS iterations.
        device (str): Target device (e.g., 'cuda').

    Returns:
        List of [A, B, C] factor matrices (torch.Tensor) per tensor, all on the specified device.
    """
    factors_list = []

    for tensor in tensor_list:
        if not tensor.is_sparse or tensor.dim() != 3:
            raise ValueError("Each tensor must be a 3D sparse tensor in COO format.")

        tensor = tensor.coalesce().to(device)  # Ensure tensor is on GPU and coalesced

        A, B, C = sparse_parafac_gpu(tensor, rank=rank, n_iter=n_iter, device=device)

        factors_list.append([A, B, C])

    return factors_list


# # Create a sparse 3D tensor on GPU
# I, J, K = 50, 60, 70
# nnz = 500

# i = torch.randint(0, I, (nnz,))
# j = torch.randint(0, J, (nnz,))
# k = torch.randint(0, K, (nnz,))
# v = torch.randn(nnz)

# indices = torch.stack([i, j, k])
# values = v

# sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(I, J, K)).coalesce().to('cuda')

# # Run sparse PARAFAC
# A, B, C = sparse_parafac_gpu(sparse_tensor, rank=10, n_iter=10)
