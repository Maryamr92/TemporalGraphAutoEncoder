# utils.py

import torch
from ModelWrapper import ModelWrapper

def prepare_wrapped_model(model, input_dim, nnodes, Time, Rank, device="cpu"):
    """
    Prepare a wrapped model for summary() or evaluation.

    Args:
        model (nn.Module): Your TGCN_Autoencoder model.
        input_dim (int): Input feature dimension (F).
        nnodes (int): Number of nodes (n).
        Time (int): Number of time steps (T).
        Rank (int): Decomposition rank (Rank).
        device (str): "cpu" or "cuda".

    Returns:
        wrapped_model (nn.Module): Model wrapped to accept only (input) for forward().
        input_tensor (torch.Tensor): Dummy input tensor matching input size.
    """
    input_shape = (nnodes, Time, input_dim)  # (n, T, F)

    # Create dummy inputs
    input_tensor = torch.randn(input_shape).to(device)
    A = torch.rand(nNodes, Rank).to(device)
    B = torch.rand(nNodes, Rank).to(device)
    C = torch.rand(Time, Rank).to(device)

    # Wrap the model
    wrapped_model = ModelWrapper(model, A, B, C, Rank).to(device)

    return wrapped_model, input_tensor
