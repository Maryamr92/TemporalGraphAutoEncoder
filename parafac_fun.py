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


### === DENSE PARAFAC === ###
def parafac_decomposition_dense(tensor, rank, n_iter_max=100):
    """
    Apply dense PARAFAC decomposition to a 3D tensor.

    Args:
        tensor (np.ndarray): Input dense tensor.
        rank (int): Number of components.
        n_iter_max (int): Max number of iterations.

    Returns:
        A, B, C (torch.Tensor): Factor matrices.
    """
    tensor = np.squeeze(tensor)
    tensor = tl.tensor(tensor)

    weights, factors = decom.parafac(
        tensor,
        rank=rank,
        normalize_factors=True,
        n_iter_max=n_iter_max,
        tol=1e-6,
        init='svd'
    )
    # print(f'weight dense, {weights}')

    # Sort by descending weights
    sorted_idx = np.argsort(-weights)
    sorted_factors = [f[:, sorted_idx] for f in factors]

    sorted_weights =  weights[sorted_idx]

    # print(f'sorted_weights dense, {sorted_weights}')

    A, B, C = [torch.from_numpy(f) for f in sorted_factors]

    # print(f'A Dense matrix,  {A}')
    # print(f'B Dense matrix,  {B}')
    # print(f'C Dense matrix,  {C}')

    reconstructed_tensor = cp_to_tensor((weights, factors))
    error = tl.norm(tensor - reconstructed_tensor)

    # print("Reconstruction Error dense (Frobenius norm):", error)

    return A, B, C


def parafac_decomposition_list_dense(tensor_list, rank):
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

        A, B, C = parafac_decomposition_dense(tensor, rank)

        factors_list.append([A, B, C])

    return factors_list


### === SPARSE PARAFAC === ###
def parafac_decomposition_sparse(tensor, rank, n_iter_max=100):
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
    dense_tensor = tensor.numpy()

    # Get coordinates of nonzero elements
    coords = np.array(np.nonzero(dense_tensor))  # shape (ndim, nnz)

    # Get data (values) at those coords
    data = dense_tensor[tuple(coords)]

    # Create sparse COO tensor
    T_sparse = sparse.COO(coords, data, shape=dense_tensor.shape)

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
    tensorly_tensor = tl.tensor(tensor, dtype='float')

    error = tl.norm(tensorly_tensor - reconstructed_tensor)

    # print("Reconstruction Error sparse (Frobenius norm):", error)



    return A, B, C


def parafac_decomposition_list_sparse(tensor_list, rank):
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

        A, B, C = parafac_decomposition_sparse(tensor, rank)
        factors_list.append([A, B, C])

    return factors_list

