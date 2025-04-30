
import torch
import torch.nn as nn
import torch.nn.functional as F
import Encoder
import Decoder



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

