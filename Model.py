
import torch
import torch.nn as nn
import torch.nn.functional as F
import Encoder
import Decoder

from torchinfo import summary


class TGCN_Autoencoder(nn.Module):
    def __init__(self, input_X, layer_dims, nNodes, Time, Rank, dropout=0.0):
        """
        Args:
            layer_dims (list): List of dimensions, e.g., [input_dim, hidden1, hidden2, ..., output_dim]
            nNodes (int): Number of nodes
            Time (int): Number of time steps
            Rank (int): Rank for decomposition
            dropout (float): Dropout probability
        """
        super(TGCN_Autoencoder, self).__init__()

        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.n_layers = len(layer_dims)   # Number of Encoder layers

        # Build Encoder layers dynamically
        for i in range(self.n_layers):
            self.layers.append(
                Encoder.TEncoder(
                    M_in= input_X if i == 0 else layer_dims[i-1],
                    M_out=layer_dims[i],
                    nNodes=nNodes,
                    Time=Time,
                    Rank=Rank
                )
            )

        # Decoder input comes from the last encoder output
        self.decoder = Decoder.TDecoder(
            input_shape=layer_dims[-1],
            nNodes=nNodes,
            Time=Time,
            Rank=Rank
        )

    def forward(self, input, A, B, C, Rank):
        x = input

        # Pass through all encoder layers
        for idx, layer in enumerate(self.layers):
            x = layer(x, A, B, C, Rank)
            if idx != self.n_layers:
                x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Decode
        A_hat, B_hat, C_hat = self.decoder(x)

        return [A_hat, B_hat, C_hat]


# ----------------- Wrapper for Model to Fix Extra Inputs -----------------
class ModelWrapper(nn.Module):
    def __init__(self, model, A, B, C, Rank):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.A = A
        self.B = B
        self.C = C
        self.Rank = Rank

    def forward(self, x):
        return self.model(x, self.A, self.B, self.C, self.Rank)

# ----------------- Utility Function: Prepare Wrapped Model -----------------
def prepare_wrapped_model(model, nNodes, Time, Rank, device="cpu"):
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

    A = torch.rand(nNodes, Rank)
    B = torch.rand(nNodes, Rank)
    C = torch.rand(Time, Rank)

    # Wrap the model
    wrapped_model = ModelWrapper(model, A, B, C, Rank)

    return wrapped_model


# ----------------- Utility Function: Show Model Summary -----------------
def show_model_summary(wrapped_model, nNodes, Time, Features):
    """
    Display a nice summary of the model.

    Args:
        wrapped_model (nn.Module): Wrapped model ready for summary.
        input_tensor (torch.Tensor): Dummy input.
    """
    input_shape= (nNodes, Time, Features)
    input_tensor = torch.randn(input_shape)

    summary(
        wrapped_model,
        input_size=input_tensor.shape,
        verbose=1,
        col_names=["input_size", "output_size", "num_params", "trainable"],
    )

