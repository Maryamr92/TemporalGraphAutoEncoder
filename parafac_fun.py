import torch
import numpy as np
import tensorly as tl
import tensorly.decomposition as decom
from tensorly.cp_tensor import cp_to_tensor
from tensorly.contrib.sparse import tensor as sparse_tensor



def parafac_gen(adj_tensor, Rank):

    if not tl.is_tensor(adj_tensor):
        adj_tensor = tl.tensor(adj_tensor)
        adj_tensor = np.squeeze(adj_tensor)

    # adj_tensor = tl.tensor(adj_tensor)
    # adj_tensor = np.squeeze(adj_tensor)

    # Decompose the adjacency tensor into factors A, B, C
    weights, factors = decom.parafac(adj_tensor, rank=Rank, normalize_factors=True,
                                     n_iter_max=100,  # you can set this higher for better convergence
                                     tol=1e-6,  # stopping threshold
                                     init='svd',  # or 'random'
                                     verbose=0
                                     )

    # print(f"weights, {weights}")

    # Extract the factor matrices (A, B, C) and append to the factors_list
    # A, B, C = factors[0], factors[1], factors[2]

    sorted_indices = np.argsort(-weights)  # Negative sign for descending order
    sorted_weights = weights[sorted_indices]
    sorted_factors = [factor[:, sorted_indices] for factor in factors]
    # print(f"sorted_weights, {sorted_weights}")
    # print(f"sorted_factors, {sorted_factors}")

    A, B, C = sorted_factors[0], sorted_factors[1], sorted_factors[2]



    # print(f"factors, {factors}")

    # print(f"A, {A}")
    # print(f"B, {B}")
    # print(f"C, {C}")

    list_errors = []
    adj_tensor_np = tl.to_numpy(adj_tensor)

    # for r in range(1, Rank + 1):
    #     partial_weights = sorted_weights[:r]
    #     partial_factors = [f[:, :r] for f in (A, B, C)]
    #     reconstructed = cp_to_tensor((partial_weights, partial_factors))
    #
    #     print(f"partial_weights, {partial_weights}")
    #
    #     error = np.linalg.norm(adj_tensor_np - reconstructed)
    #     list_errors.append(error)

    # print(f"Original tensor slice: {adj_tensor_np[:5, :3, 0]}")
    # print(f"Reconstructed tensor (rank {Rank}): {reconstructed[:5, :3, 0]}")
    # print(f"Reconstruction errors for ranks 1 to {Rank}: {list_errors}")

    # input('---')

    # factors_list.append([A, B, C])
    # print(f"weights, {weights}")

    A = torch.from_numpy(A)
    B = torch.from_numpy(B)
    C = torch.from_numpy(C)

    return A, B, C


# Decomposition wrapper for a list of adjacency tensors
def parafac_gen_list(adj_tensor_list, Rank):
    factors_list = []

    for adj_tensor in adj_tensor_list:

        adj_tensor = tl.tensor(adj_tensor)

        # indices = adj_tensor['indices']
        # values = adj_tensor['values']
        # shape = adj_tensor['shape']
        #
        # # Convert torch tensors to numpy if needed
        # if torch.is_tensor(indices):
        #     indices = indices.cpu().numpy()
        # if torch.is_tensor(values):
        #     values = values.cpu().numpy()

        # Convert to tensorly sparse tensor
        # tl_tensor = sparse_tensor((values, indices), shape)

        ## Decompose using CP/PARAFAC
        # weights, factors = decom.parafac(tl_tensor, rank=Rank, normalize_factors=True,
        #                                  n_iter_max=100, tol=1e-6,
        #                                  init='svd', verbose=0)


        # Decompose the tensor into factors A, B, C
        # Decompose the adjacency tensor into factors A, B, C
        weights, factors = decom.parafac(adj_tensor, rank=Rank, normalize_factors=True,
                                         n_iter_max=100,  # you can set this higher for better convergence
                                         tol=1e-6,  # stopping threshold
                                         init='svd',  # or 'random'
                                         verbose=0
                                         )

        sorted_indices = np.argsort(-weights)  # Negative sign for descending order
        sorted_weights = weights[sorted_indices]
        sorted_factors = [factor[:, sorted_indices] for factor in factors]

        # Extract the factor matrices (A, B, C) and append to the factors_list
        A, B, C = sorted_factors[0], sorted_factors[1], sorted_factors[2]

        A = torch.from_numpy(A)
        B = torch.from_numpy(B)
        C = torch.from_numpy(C)

        factors_list.append([A, B, C])


    return factors_list