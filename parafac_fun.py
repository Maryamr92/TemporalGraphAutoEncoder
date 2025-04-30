
import tensorly as tl
import tensorly.decomposition as decom
from tensorly.cp_tensor import cp_to_tensor



def parafac_gen(adj_tensor, Rank):
    adj_tensor = tl.tensor(adj_tensor)
    # Decompose the adjacency tensor into factors A, B, C
    weights, factors = decom.parafac(adj_tensor, rank=Rank)

    # Extract the factor matrices (A, B, C) and append to the factors_list
    A, B, C = factors[0], factors[1], factors[2]
    # factors_list.append([A, B, C])

    return A, B, C


# Decomposition wrapper for a list of adjacency tensors
def parafac_gen_list(adj_tensor_list, Rank):
    factors_list = []

    for adj_tensor in adj_tensor_list:
        adj_tensor = tl.tensor(adj_tensor)

        # Decompose the tensor into factors A, B, C
        weights, factors = decom.parafac(adj_tensor, rank=Rank)

        # Extract the factor matrices and append to list
        A, B, C = factors[0], factors[1], factors[2]
        # A, B, C = A.reshape(40, 1), B.reshape(40, 1), C.reshape(40, 1)
        factors_list.append([A, B, C])

    return factors_list