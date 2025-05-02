import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------- Decoder Layer --------------------
class TDecoder(nn.Module):
    def __init__(self, input_shape, nNodes, Time, Rank):
        super(TDecoder, self).__init__()
        self.m1, self.m2, self.m3 = input_shape
        self.rank = Rank
        self.nodes = nNodes
        self.Time = Time

        self.theta1 = nn.Parameter(torch.randn(self.rank, self.m1))
        self.alpha1 = nn.Parameter(torch.randn(self.m2, self.nodes))

        self.theta2 = nn.Parameter(torch.randn(self.rank, self.m1))
        self.alpha2 = nn.Parameter(torch.randn(self.m2, self.nodes))

        self.theta3 = nn.Parameter(torch.randn(self.rank, self.m1))
        self.alpha3 = nn.Parameter(torch.randn(self.m2, self.Time))

        self.reset_parameters()

        # print(f"theta1, {self.theta1}")
        # print(f"input_shape, {input_shape}")

    def reset_parameters(self):
        for param in [self.theta1, self.theta2, self.theta3, self.alpha1, self.alpha2, self.alpha3]:
            stdv = 1. / (param.size(1) ** 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, Z):

        A_hat = torch.zeros(self.rank, self.nodes, device=Z.device)
        B_hat = torch.zeros(self.rank, self.nodes, device=Z.device)
        C_hat = torch.zeros(self.rank, self.Time, device=Z.device)

        # print(f"Z_shape, {Z.shape}")
        # print(f"self.m3_shape, {self.m3}")

        if Z.dim() == 2:
            Z = Z.unsqueeze(-1)  # Add third dimension → (B, F, 1)

        # Apply projection to recover factor matrices
        for j in range(self.m3):
            Z_j = Z[:, :, j]

            # ZZ_T = Z_j @ Z_j.T

            A_ = self.theta1 @ Z_j @ self.alpha1
            B_ = self.theta2 @ Z_j @ self.alpha2
            C_ = self.theta3 @ Z_j @ self.alpha3

            A_hat += A_
            B_hat += B_
            C_hat += C_

        A_hat = A_hat.T  # shape (n, R)
        B_hat = B_hat.T  # shape (n, R)
        C_hat = C_hat.T  # shape (T, R)

        return A_hat, B_hat, C_hat