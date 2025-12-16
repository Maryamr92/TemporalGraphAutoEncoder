import torch
import torch.nn as nn
import torch.nn.functional as F

from Encoder import FactorEncoder
from Decoder import FactorDecoder, FeatureDecoder


class TGAutoencoder(nn.Module):
    """
    Temporal Graph Autoencoder composed of FactorEncoders (A, B, C)
    and decoders for factors + feature reconstruction.
    """

    def __init__(self, X_size, A_size, B_size, C_size, layer_dims, device=None):
        """
        layer_dims = [
            (dim_A_encoder_1, dim_B_encoder_1, dim_C_encoder_1),
            (dim_A_encoder_2, dim_B_encoder_2, dim_C_encoder_2),
            ...
        ]
        """
        super(TGAutoencoder, self).__init__()

        self.X_size = X_size
        self.A_size = A_size
        self.B_size = B_size
        self.C_size = C_size

        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims)

        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        ################################################
        # ------------------- Encoders -----------------
        ################################################

        # Encoder for A
        self.layers_A = nn.ModuleList()
        self.layers_A.append(
            FactorEncoder(
                size_in=X_size,
                size_out=(layer_dims[0][0], A_size[1], X_size[2]),
                size_M=A_size
            )
        )

        for i in range(1, self.n_layers):
            self.layers_A.append(
                FactorEncoder(
                    size_in=(layer_dims[i - 1][0], A_size[1], X_size[2]),
                    size_out=(layer_dims[i][0], A_size[1], X_size[2]),
                    size_M=A_size
                )
            )

        # Encoder for B
        self.layers_B = nn.ModuleList()
        self.layers_B.append(
            FactorEncoder(
                size_in=X_size,
                size_out=(layer_dims[0][1], B_size[1], X_size[2]),
                size_M=B_size
            )
        )

        for i in range(1, self.n_layers):
            self.layers_B.append(
                FactorEncoder(
                    size_in=(layer_dims[i - 1][1], B_size[1], X_size[2]),
                    size_out=(layer_dims[i][1], B_size[1], X_size[2]),
                    size_M=B_size
                )
            )

        # Encoder for C
        self.layers_C = nn.ModuleList()
        self.layers_C.append(
            FactorEncoder(
                size_in=X_size,
                size_out=(layer_dims[0][2], C_size[1], X_size[2]),
                size_M=C_size
            )
        )

        for i in range(1, self.n_layers):
            self.layers_C.append(
                FactorEncoder(
                    size_in=(layer_dims[i - 1][2], C_size[1], X_size[2]),
                    size_out=(layer_dims[i][2], C_size[1], X_size[2]),
                    size_M=C_size
                )
            )

        ################################################
        # ------------------- Decoders -----------------
        ################################################

        # Decoders for A, B, C
        self.decoder_A = FactorDecoder(
            size_Z=(layer_dims[-1][0], A_size[1], X_size[2]),
            size_M=A_size
        )

        self.decoder_B = FactorDecoder(
            size_Z=(layer_dims[-1][1], B_size[1], X_size[2]),
            size_M=B_size
        )

        self.decoder_C = FactorDecoder(
            size_Z=(layer_dims[-1][2], C_size[1], X_size[2]),
            size_M=C_size
        )

        # Decoder for X (concatenated latent factors)
        cat_dim = layer_dims[-1][0] + layer_dims[-1][1] + layer_dims[-1][2]
        self.decoder_X = FeatureDecoder(
            size_Z=(cat_dim, C_size[1], X_size[2]),
            size_X=X_size
        )

    ################################################
    # ------------------- Reset --------------------
    ################################################

    def reset(self):
        """Reset all learnable layers to initial state."""

        for layerA in self.layers_A:
            if hasattr(layerA, 'reset_parameters'):
                layerA.reset_parameters()

        for layerB in self.layers_B:
            if hasattr(layerB, 'reset_parameters'):
                layerB.reset_parameters()

        for layerC in self.layers_C:
            if hasattr(layerC, 'reset_parameters'):
                layerC.reset_parameters()

        if hasattr(self.decoder_A, 'reset_parameters'):
            self.decoder_A.reset_paramete
