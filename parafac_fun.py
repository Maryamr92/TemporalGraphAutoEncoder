import torch
import numpy as np
import tensorly as tl
import tensorly.decomposition as decom
from tensorly.cp_tensor import cp_to_tensor
# from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac
# import sparse  # Sparse array library, NOT PyTorch
# import tensorly.contrib.sparse as stl
import csv
from tensorly.contrib.sparse import tensor as stl_tensor
from tensorly.decomposition import parafac as sparse_parafac

# tl.set_backend('numpy')  # Can be changed to 'pytorch' if needed
tl.set_backend('pytorch')


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

def parafac_decomposition_list_sparse(csv_file, time_start, time_end, nNodes, rank, device=None, save_path=None):
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

    # Containers for sparse tensor indices
    src_list = []
    dst_list = []
    time_list = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            src, dst, t = map(int, row)
            if time_start <= t <= time_end:
                src_list.append(int(src))
                dst_list.append(int(dst))
                time_list.append(int(t))

    Time = time_end - time_start

    # Convert to tensor indices (3 x N)
    indices = torch.tensor([src_list, dst_list, time_list], dtype=torch.long)
    values = torch.ones(len(src_list))  # All edge values = 1

    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(nNodes, nNodes, Time))

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


def load_edges_(csv_path, time_start, time_end, nNodes):
    """
    Load edges from CSV and return indices, values, and shape — all in pure Python.

    Returns:
        indices: tuple of three lists (src, dst, time)
        values: list of float (edge weights)
        shape: 3D tensor shape tuple
    """
    src_list = []
    dst_list = []
    time_list = []

    with open(csv_path, 'r') as f:
        for line in f:
            src, dst, t = map(int, line.strip().split(','))
            if time_start <= t <= time_end:
                src_list.append(src)
                dst_list.append(dst)
                time_list.append(t - time_start)  # Normalize time to start at 0

    if not src_list:
        raise ValueError("No edges found in the given time window.")

    indices = (src_list, dst_list, time_list)
    values = [1.0] * len(src_list)
    shape = (nNodes, nNodes, time_end - time_start + 1)

    return indices, values, shape


def run_sparse_parafac(csv_path, time_start, time_end, nNodes, rank=10, n_iter_max=100):
    """
    Run sparse PARAFAC on a 3D sparse tensor using only Python (via TensorLy).

    Returns:
        weights: list of floats
        factors: list of 2D lists (A, B, C)
    """
    indices, values, shape = load_edges_(csv_path, time_start, time_end, nNodes)
    sparse_tensor = stl_tensor((indices, values), shape)
    weights, factors = sparse_parafac(
        sparse_tensor,
        rank=rank,
        n_iter_max=n_iter_max,
        normalize_factors=True,
        init='random'
    )

    sorted_idx = sorted(range(len(weights)), key=lambda i: -weights[i])
    sorted_weights = [weights[i] for i in sorted_idx]

    sorted_factors = []
    for factor in factors:
        # Transpose, reorder columns, then transpose back
        factor_T = list(zip(*factor))
        sorted_T = [factor_T[i] for i in sorted_idx]
        sorted_matrix = list(zip(*sorted_T))
        sorted_factors.append([list(row) for row in sorted_matrix])

    # # Convert to PyTorch tensors
    # A, B, C = [torch.tensor(f, dtype=torch.float32) for f in sorted_factors]
    # weights_tensor = torch.tensor(sorted_weights, dtype=torch.float32)

    print(f'Done !! with tensor shape : {shape}')

    return sorted_weights, sorted_factors

