import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import TEncoder
from Decoder import TDecoder
from Data_Generator import split_sum_into_3d


# -------------------- Full TGCN Model (Encoder + Decoder) --------------------
class TGCN_Autoencoder(nn.Module):

    def __init__(self, input_X, layer_dims, nNodes, Time, Rank, dropout=0.0, device=None):

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert layer_dims integers into 3D tuples for all layers
        layer_dims_3d = [split_sum_into_3d(dim) for dim in layer_dims]

        super(TGCN_Autoencoder, self).__init__()

        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.n_layers = len(layer_dims)  # Number of Encoder layers

        # Build Encoder layers dynamically
        for i in range(self.n_layers):
            self.layers.append(
                TEncoder(
                    M_in=input_X if i == 0 else [layer_dims[i - 1], Rank, input_X[2]],
                    M_out=layer_dims_3d[i],
                    nNodes=nNodes,
                    Time=Time,
                    Rank=Rank,
                    device=self.device
                )
            )
        # print(f"input_X, {input_X[2]}")

        # Decoder input comes from the last encoder output
        self.decoder = TDecoder(
            input_shape=(layer_dims[-1], Rank, input_X[2]),
            nNodes=nNodes,
            Time=Time,
            Rank=Rank,
            device=self.device)

    def reset(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if hasattr(self.decoder, 'reset_parameters'):
            self.decoder.reset_parameters()

    def forward(self, input, A, B, C, Rank):
        x = input

        # Pass through all encoder layers
        for idx, layer in enumerate(self.layers):
            x = layer(x, A, B, C, Rank)
            if idx != self.n_layers - 1:  # last layer no relu
                x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        A_hat, B_hat, C_hat = self.decoder(x)

        return [A_hat, B_hat, C_hat]

