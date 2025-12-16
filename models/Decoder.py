import torch
import torch.nn as nn


########################################################
# -------------------- Factor Decoder ------------------
########################################################

class FactorDecoder(nn.Module):
    def __init__(self, size_Z, size_M, device=None):
        super(FactorDecoder, self).__init__()
        self.size_Z = size_Z
        self.size_M = size_M
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Learnable decoding matrices
        self.theta = nn.Parameter(
            torch.randn(self.size_M[0], self.size_Z[0], device=self.device)
        )
        self.b = nn.Parameter(
            torch.randn(self.size_M[0], self.size_M[1], device=self.device)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform for weights and uniform for biases."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                stdv = 1.0 / (param.size(0) ** 0.5)
                param.data.uniform_(-stdv, stdv)

    def forward(self, Z):
        """
        Decode latent tensor Z into matrix M_hat.
        Z: (size_Z[0], size_Z[1], size_Z[2])
        """
        M_hat = torch.zeros(self.size_M[0], self.size_M[1], device=self.device)

        for i in range(self.size_Z[2]):
            Z_i = Z[:, :, i]
            P = self.theta @ Z_i
            M_hat += P + self.b

        return M_hat


########################################################
# ------------------- Feature Decoder ------------------
########################################################

class FeatureDecoder(nn.Module):
    def __init__(self, size_Z, size_X, device=None):
        super(FeatureDecoder, self).__init__()
        self.size_Z = size_Z
        self.size_X = size_X
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Learnable decoding matrices
        self.theta = nn.Parameter(
            torch.randn(self.size_X[0], self.size_Z[0], device=self.device)
        )
        self.alpha = nn.Parameter(
            torch.randn(self.size_Z[1], self.size_X[1], device=self.device)
        )
        self.b = nn.Parameter(
            torch.randn(self.size_X[0], self.size_X[1], device=self.device)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform for weights and uniform for biases."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                stdv = 1.0 / (param.size(0) ** 0.5)
                param.data.uniform_(-stdv, stdv)

    def forward(self, Z):
        """
        Decode latent tensor Z into feature tensor X_hat.
        Z: (size_Z[0], size_Z[1], size_Z[2])
        """
        X_hat = torch.zeros(
            self.size_X[0], self.size_X[1], self.size_X[2], device=self.device
        )

        for i in range(self.size_Z[2]):
            Z_i = Z[:, :, i]
            P = Z_i @ self.alpha
            P = self.theta @ P
            X_hat[:, :, i] = P + self.b

        return X_hat
