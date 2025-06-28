
import torch
import torch.nn as nn
from torchinfo import summary

# ----------------- Wrapper for Model to Fix Extra Inputs -----------------
class ModelWrapper(nn.Module):
    def __init__(self, model, A, B, C, Rank, device=None):
        super(ModelWrapper, self).__init__()
        self.model = model.to(device) if device else model
        self.A = A
        self.B = B
        self.C = C
        self.Rank = Rank

    def forward(self, x):
        return self.model(x, self.A, self.B, self.C, self.Rank)

# ----------------- Utility Function: Prepare Wrapped Model -----------------
def prepare_wrapped_model(model, nNodes, Time, Rank, device=None):
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

    # Create dummy inputs

    A = torch.rand(nNodes, Rank, device=device)
    B = torch.rand(nNodes, Rank, device=device)
    C = torch.rand(Time, Rank, device=device)
    model = model.to(device)

    # Wrap the model
    wrapped_model = ModelWrapper(model, A, B, C, Rank).to(device)

    return wrapped_model


# ----------------- Utility Function: Show Model Summary -----------------
def show_model_summary(wrapped_model, nNodes, Time, Features, device=None):
    """
    Display a nice summary of the model.

    Args:
        wrapped_model (nn.Module): Wrapped model ready for summary.
        input_tensor (torch.Tensor): Dummy input.
    """
    input_shape= (nNodes, Time, Features)
    input_tensor = torch.randn(input_shape, device=device)

    summary(
        wrapped_model,
        input_size=input_tensor.shape,
        verbose=1,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        device=device
    )

