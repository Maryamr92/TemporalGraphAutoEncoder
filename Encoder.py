import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# -------------------- Encoder Layer (TEncoder) --------------------
class TEncoder(nn.Module):
    def __init__(self, M_in, M_out, nNodes, Time, Rank):
        super(TEncoder, self).__init__()
        self.m1_in, self.m2_in, self.m3_in = M_in
        self.m1_out, self.m2_out, self.m3_out = M_out
        self.nodes = nNodes
        self.rank = Rank
        self.Time = Time
        # self.dropout = dropout

        # Learnable projection matrices
        self.W1 = nn.Parameter(torch.randn(self.m1_out, self.m1_in))
        self.V1 = nn.Parameter(torch.randn(self.m2_in, self.nodes))

        self.W2 = nn.Parameter(torch.randn(self.m2_out, self.m1_in))
        self.V2 = nn.Parameter(torch.randn(self.m2_in, self.nodes))

        self.W3 = nn.Parameter(torch.randn(self.m3_out, self.m1_in))
        self.V3 = nn.Parameter(torch.randn(self.m2_in, self.Time))

        self.reset_parameters()

    def reset_parameters(self):
        for param in [self.W1, self.V1, self.W2, self.V2, self.W3, self.V3]:
            stdv = 1. / (param.size(1) ** 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, A, B, C, Rank):

        output_R = torch.zeros(self.m1_out, self.m2_out, self.m3_out, device=input.device)

        for i in range(self.m3_in):
            H_i = input[:, :, i]  # (m1, m2)
            P1 = torch.mm(self.W1, H_i) @ self.V1  # (m1_out, nodes)
            P2 = torch.mm(self.W2, H_i) @ self.V2  # (m2_out, nodes)
            P3 = torch.mm(self.W3, H_i) @ self.V3  # (m3_out, Time)

            for r in range(Rank):
                part1 = P1 @ A[:, r].unsqueeze(1)  # (m1_out, 1)
                part2 = P2 @ B[:, r].unsqueeze(1)  # (m2_out, 1)
                part3 = P3 @ C[:, r].unsqueeze(1)  # (m3_out, 1)

                rank1_tensor = (part1 @ part2.T).unsqueeze(2) * part3.T
                output_R += rank1_tensor

        return output_R