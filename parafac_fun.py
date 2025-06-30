import torch
import numpy as np
import tensorly as tl
import tensorly.decomposition as decom
from tensorly.cp_tensor import cp_to_tensor
# from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac
# import tensorly.contrib.sparse as stl
import csv
from tensorly.contrib.sparse import tensor as stl_tensor
from tensorly.decomposition import parafac as sparse_parafac
import sparse         # Sparse array library, NOT PyTorch
import pickle

tl.set_backend('numpy')



### === DENSE PARAFAC === ###
def parafac_decomposition_dense(adj_tensor, rank, n_iter_max=100, device=None, save_path_prefix=None):
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

    sorted_weights = weights[sorted_idx]

    # print(f'sorted_weights dense, {sorted_weights}')
    A, B, C = sorted_factors

    if save_path_prefix:
        torch.save(
            {
                "A": A,
                "B": B,
                "C": C,
                "weights": torch.tensor(sorted_weights, dtype=torch.float32)
            },
            f"{save_path_prefix}_factors.pt"
        )

    reconstructed_tensor = cp_to_tensor((weights, factors))
    error = tl.norm(adj_tensor - reconstructed_tensor)

    # print("Reconstruction Error dense (Frobenius norm):", error)

    return A.to(device), B.to(device), C.to(device), sorted_weights.to(device)


def parafac_decomposition_list_dense(tensor_list, rank, device=None, save_prefix=None):
    """
    Apply dense PARAFAC to a list of tensors.

    Args:
        tensor_list (list of np.ndarray): List of 3D tensors.
        rank (int): Rank of decomposition.

    Returns:
        list of [A, B, C] factors (torch.Tensors)
    """

    factors_list = []

    results = []
    for i, tensor in enumerate(tensor_list):
        prefix = f"{save_prefix}_sample{i}" if save_prefix else None
        A, B, C, weights = parafac_decomposition_dense(tensor, rank, device=device, save_path_prefix=prefix)
        results.append((A, B, C, weights))
    return results


### === SPARSE PARAFAC === ###
def parafac_decomposition_sparse(adj_tensor, rank, n_iter_max=100, device=None):
    """
    Apply sparse PARAFAC directly on a PyTorch sparse COO tensor (3D).

    Args:
        adj_tensor (torch.sparse_coo_tensor): Sparse tensor in COO format.
        rank (int): Rank of decomposition.
        n_iter_max (int): Max number of iterations.
        device (torch.device or None): Device to move result tensors.

    Returns:
        A, B, C (torch.Tensor): Factor matrices.
        sorted_weights (torch.Tensor): Sorted weights from CP decomposition.
    """
    # Ensure tensor is coalesced
    adj_tensor = adj_tensor.coalesce()

    # Extract indices and values
    indices = adj_tensor.indices().cpu().numpy()  # shape [3, nnz]
    values = adj_tensor.values().cpu().numpy()
    shape = adj_tensor.shape

    # Create tensorly sparse tensor
    tensorly_sparse = stl_tensor((indices, values), shape=shape)

    # Run sparse PARAFAC decomposition
    weights, factors = sparse_parafac(
        tensorly_sparse,
        rank=rank,
        n_iter_max=n_iter_max,
        normalize_factors=True,
        init='random'
    )

    # Sort by weight
    weights = np.array(weights)
    sorted_idx = np.argsort(-weights)
    sorted_factors = [f[:, sorted_idx] for f in factors]
    sorted_weights = weights[sorted_idx]

    # Convert to PyTorch tensors
    A, B, C = [torch.tensor(f, dtype=torch.float32) for f in sorted_factors]
    weights_tensor = torch.tensor(sorted_weights, dtype=torch.float32)

    return A.to(device), B.to(device), C.to(device), weights_tensor.to(device)

def parafac_decomposition_list_sparse(tensor_list, rank, device=None, save_path=None):
    """
    Apply sparse PARAFAC decomposition to a list of sparse tensors and save all results together.

    Args:
        tensor_list (list of torch.sparse_coo_tensor): Sparse tensors to decompose.
        rank (int): Rank of decomposition.
        device (str or torch.device): Target device.
        save_path (str): Path to save the .pt file with all decomposed factors.

    Returns:
        List of dictionaries with 'A', 'B', 'C', 'weights'.
    """

    # Create sparse tensor
    results = []
    for i, tensor in enumerate(tensor_list):
        prefix = f"parafac_sample{i}" if save_prefix else None
        A, B, C, weights = parafac_decomposition_sparse(tensor, rank, device=device, save_path_prefix=prefix)
        results.append((A, B, C, weights))
    return results

    A, B, C, weights = parafac_decomposition_sparse(sparse_tensor, rank, device=device)

    results = [{
            "A": A,
            "B": B,
            "C": C,
            "weights": weights
        }]

    if save_path:
        torch.save(results, save_path)
        print(f"Saved parafac decompositions to {save_path}")

    return results

def load_edges_numpy(csv_path, time_start, time_end, nNodes):
    data = np.loadtxt(csv_path, delimiter=',', dtype=int)
    mask = (data[:, 2] >= time_start) & (data[:, 2] < time_end)
    data = data[mask]

    # Filter by node index range
    node_mask = (data[:, 0] < nNodes) & (data[:, 1] < nNodes)
    filtered = data[node_mask]

    src_list = filtered[:, 0].tolist()
    dst_list = filtered[:, 1].tolist()
    time_list = (filtered[:, 2] - time_start).tolist()

    return src_list, dst_list, time_list


def load_edges_(csv_path, time_start, time_end, nNodes):
    """
    Load edges from CSV and return indices, values, and shape — all in pure Python.

    Returns:
        indices: tuple of three lists (src, dst, time)
        values: list of float (edge weights)
        shape: 3D tensor shape tuple
    """
    src_list, dst_list, time_list =  load_edges_numpy(csv_path, time_start, time_end, nNodes)
    # Stack the coordinate lists into a 2D NumPy array of shape (3, n_nonzero)
    indices = np.stack([src_list, dst_list, time_list])
    values = np.ones(len(src_list))  # use np.ones for a proper ndarray
    shape = (nNodes, nNodes, (time_end - time_start))

    print(f"indices shape: {indices.shape}, values shape: {values.shape}, shape: {shape}")
    print(type(indices), indices.shape)
    print(type(values), values.shape)
    print(shape)

    return indices, values, shape


def run_sparse_parafac(csv_path, time_start, time_end, nNodes, rank=10, n_iter_max=20):
    """
    Run sparse PARAFAC on a 3D sparse tensor using only Python (via TensorLy).

    Returns:
        weights: list of floats
        factors: list of 2D lists (A, B, C)
    """
    indices, values, shape = load_edges_(csv_path, time_start, time_end, nNodes)

    print("Max src:", np.max(indices[0]))
    print("Max dst:", np.max(indices[1]))
    print("Max time:", np.max(indices[2]))

    sparse_tensor = sparse.COO(coords=indices, data=values, shape=shape)

    weights, factors = sparse_parafac(
        sparse_tensor,
        rank=rank,
        n_iter_max=n_iter_max,
        normalize_factors=True,
        init='random'
    )

    print(f"weights, {weights.shape}, factors, {factors}")

    # Sort weights in descending order
    weights = np.array(weights)
    sorted_idx = np.argsort(-weights)  # Indices that would sort weights descending

    # Apply the same order to factors
    sorted_weights = weights[sorted_idx]
    sorted_factors = [f[:, sorted_idx] for f in factors]

    # Save as np array
    with open('factors-200-300.pkl', 'wb') as f:
        pickle.dump({'factors': sorted_factors, 'weights': sorted_weights}, f)

    # # Convert each factor matrix from numpy to torch tensor
    # torch_factors = [torch.tensor(factor, dtype=torch.float32) for factor in sorted_factors]
    #
    # # Save tensors (example: save as a dictionary)
    # torch.save({
    #     'weights': torch.tensor(sorted_weights, dtype=torch.float32),
    #     'factors': torch_factors
    # }, 'model_factors.pt')

    print(f'Done !! with tensor shape : {shape}')

    return sorted_weights, sorted_factors

