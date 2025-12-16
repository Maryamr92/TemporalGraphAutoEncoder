import torch
import torch.nn as nn
import torch.nn.functional as F


####################################################################
# -------------------- Encoder Layer (FactorEncoder) ----------------
####################################################################

class FactorEncoder(nn.Module):
    def __init__(self, size_in, size_out, size_M, device=None):
        super(FactorEncoder, self).__init__()

        self.size_in = size_in      # Shape of input embedding tensor
        self.size_out = size_out    # Shape of output embedding tensor
        self.size_M = size_M

        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Layer normalization on output channels
        self.layer_norm = nn.LayerNorm(size_out[0])

        # Learnable projection matrices
        self.W = nn.Parameter(
            torch.randn(self.size_out[0], self.size_in[0], device=self.device)
        )
        self.V = nn.Parameter(
            torch.randn(self.size_in[1], self.size_M[0], device=self.device)
        )
        self.b = nn.Parameter(
            torch.randn(self.size_out[0], self.size_out[1], device=self.device)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize learnable parameters."""
        nn.init.kaiming_uniform_(self.W, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.V, a=0.01, nonlinearity='leaky_relu')

        fan_in = self.W.size(1)
        bound = 1.0 / (fan_in ** 0.5)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, input, M):
        """
        Encode input tensor using projection matrices W and V.

        Args:
            input: Tensor of shape (size_in[0], size_in[1], size_in[2])
            M:     Auxiliary matrix used in the projection

        Returns:
            embedding: Output tensor of shape (size_out[0], size_out[1], size_out[2])
        """

        embedding = torch.empty(
            self.size_out[0], self.size_out[1], self.size_out[2],
            device=self.device
        )

        for i in range(self.size_in[2]):
            H_i = input[:, :, i]

            P = self.V @ M
            P = H_i @ P
            P = self.W @ P

            embedding[:, :, i] = P + self.b

        # Nonlinearity + layer norm
        embedding = F.leaky_relu(embedding, negative_slope=0.01)

        embedding = self.layer_norm(embedding.permute(2, 1, 0))
        embedding = embedding.permute(2, 1, 0)

        return embedding
