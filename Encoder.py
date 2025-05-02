import torch
import torch.nn as nn


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

        # self.b3 = nn.Parameter(torch.randn(self.m3_out, self.Time))

        self.reset_parameters()

    def reset_parameters(self):
        for param in [self.W1, self.V1, self.W2, self.V2, self.W3, self.V3]:
            stdv = 1. / (param.size(1) ** 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, A, B, C, Rank):

        PA_list = []
        PB_list = []
        PC_list = []

        for i in range(self.m3_in):

            # print(f"self.m3_in, {self.m3_in}")

            H_i = input[:, :, i]  # (m1, m2)
            P1 = torch.mm(self.W1, H_i) @ self.V1  # (m1_out, nodes)
            P2 = torch.mm(self.W2, H_i) @ self.V2  # (m2_out, nodes)
            P3 = torch.mm(self.W3, H_i) @ self.V3  # (m3_out, Time)

            P1_list = []
            P2_list = []
            P3_list = []

            for r in range(Rank):
                part1 = P1 @ A[:, r].unsqueeze(1)  # (m1_out, 1)
                part2 = P2 @ B[:, r].unsqueeze(1)  # (m2_out, 1)
                part3 = P3 @ C[:, r].unsqueeze(1)  # (m3_out, 1)

                P1_list.append(part1)
                P2_list.append(part2)
                P3_list.append(part3)

            PA = torch.cat(P1_list, dim=1)  # (m1_out, Rank)
            PB = torch.cat(P2_list, dim=1)  # (m2_out, Rank)
            PC = torch.cat(P3_list, dim=1)  # (m3_out, Rank)

            PA_list.append(PA.unsqueeze(2))  # Add a new dimension for stacking
            PB_list.append(PB.unsqueeze(2))  # Add a new dimension for stacking
            PC_list.append(PC.unsqueeze(2))  # Add a new dimension for stacking

        # Stack all P3 slices along the 3rd dimension (m3_in)
        PA_tensor = torch.cat(PA_list, dim=1)  # Final shape: (m1_out, Rank, m3_in)
        PB_tensor = torch.cat(PB_list, dim=1)  # Final shape: (m2_out, Rank, m3_in)
        PC_tensor = torch.cat(PC_list, dim=1)  # Final shape: (m3_out, Rank, m3_in)

        PA_PB_PC = torch.cat([PA_tensor, PB_tensor, PC_tensor], dim=0)

        # print(f"PA_PB_PC, {PA_PB_PC.shape}")

        return PA_PB_PC  # Final shape: (m1_out+m2_out+m3_out+, Rank, m3_in)

