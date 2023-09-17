import torch
from torch import nn
from math import ceil

from .model import MODEL, DENOISING_MODEL
from .relu import ReLULayer

from utils.spherical_harmonics import components_from_spherical_harmonics


class SphericalHarmonicsLayer(nn.Module):
    def __init__(
        self,
        relu,
        hidden_dim,
        levels,
        time,
        omega,
        **kwargs,
    ):
        super().__init__()
        self.relu = relu
        self.levels = levels
        self.hidden_dim = hidden_dim
        # self.hidden_dim = levels**2
        self.time = time
        self.omega = omega
        self.linear = nn.Linear(levels**2, self.hidden_dim)

        if time:
            self.latents = nn.Linear(1, self.hidden_dim) # latent embedding for time
            self.linear = nn.Linear(levels**2+self.hidden_dim, self.hidden_dim) # change linear layer output dim for time

    def forward(self, input):
        out = components_from_spherical_harmonics(self.levels, input[..., :3])

        if self.time:
            time = input[..., 3:]
            time = self.latents(time)
            out = torch.cat([out, time], dim=-1)
        out = self.linear(out)
        
        if self.relu:
            return nn.functional.relu(out)
        else:
            return out


class INR(MODEL):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        hidden_layers,
        levels,
        time,
        skip,
        omega,
        relu,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time = time
        self.skip = skip
        self.hidden_layers = hidden_layers

        self.first_nonlin = SphericalHarmonicsLayer

        self.net = nn.ModuleList()
        self.net.append(self.first_nonlin(relu, hidden_dim, levels, time, omega))

        self.nonlin = ReLULayer

        for i in range(hidden_layers):
            if skip and i == ceil(hidden_layers / 2):
                self.net.append(self.nonlin(hidden_dim + input_dim, hidden_dim))
            else:
                self.net.append(self.nonlin(hidden_dim, hidden_dim))

        final_linear = nn.Linear(hidden_dim, output_dim)

        self.net.append(final_linear)

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers / 2) + 1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x


class DENOISING_INR(DENOISING_MODEL):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        hidden_layers,
        levels,
        time,
        skip,
        omega,
        relu,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time = time
        self.skip = skip
        self.hidden_layers = hidden_layers

        self.first_nonlin = SphericalHarmonicsLayer

        self.net = nn.ModuleList()
        self.net.append(self.first_nonlin(relu, hidden_dim, levels, time, omega))

        self.nonlin = ReLULayer

        for i in range(hidden_layers):
            if skip and i == ceil(hidden_layers / 2):
                self.net.append(self.nonlin(hidden_dim + input_dim, hidden_dim))
            else:
                self.net.append(self.nonlin(hidden_dim, hidden_dim))

        final_linear = nn.Linear(hidden_dim, output_dim)

        self.net.append(final_linear)

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers / 2) + 1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x
