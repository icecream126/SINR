import torch
from torch import nn
from math import ceil

from .model import MODEL, DENOISING_MODEL
from utils.posenc import SPHERE_NGP_INTERP_ENC
# 없음 ㅠㅠ

class SPHERE_NGP_INTERP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        return nn.functional.relu(self.linear(input))


class INR(MODEL):
    def __init__(
        self,
        output_dim,
        # input_dim,
        hidden_dim,
        hidden_layers,
        time,
        skip,
        finest_resolution,
        base_resolution,
        geodesic_weight,
        T,
        n_levels=2,
        n_features_per_level=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_dim = kwargs['input_dim']
        self.time = time
        self.skip = skip
        self.hidden_layers = hidden_layers
        self.posenc = SPHERE_NGP_INTERP_ENC(geodesic_weight = geodesic_weight, F = n_features_per_level,  L = n_levels, input_dim = self.input_dim, T = T, finest_resolution=finest_resolution, base_resolution = base_resolution)
        self.posenc_dim = n_levels * n_features_per_level

        # LearnableEncoding(self, lat_shape, lon_shape, mapping_size, resolution)
        self.nonlin = SPHERE_NGP_INTERP

        self.net = nn.ModuleList()
        self.net.append(self.nonlin(self.posenc_dim, hidden_dim))

        for i in range(hidden_layers):
            if skip and i == ceil(hidden_layers / 2):
                self.net.append(self.nonlin(hidden_dim + self.posenc_dim, hidden_dim))
            else:
                self.net.append(self.nonlin(hidden_dim, hidden_dim))

        final_linear = nn.Linear(hidden_dim, output_dim)

        self.net.append(final_linear)

    def forward(self, x):
        x = self.posenc(x)
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers / 2) + 1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x
