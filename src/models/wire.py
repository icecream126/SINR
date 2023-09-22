import torch
from torch import nn
from math import ceil

from .model import MODEL, DENOISING_MODEL


class GaborLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        omega,
        sigma,
        **kwargs,
    ):
        super().__init__()

        self.omega = omega
        self.sigma = sigma

        self.freqs = nn.Linear(input_dim, output_dim)
        self.scale = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        omega = self.omega * self.freqs(input)
        scale = self.scale(input) * self.sigma

        freq_term = torch.cos(omega)
        gauss_term = torch.exp(-(scale**2))
        return freq_term * gauss_term


class INR(MODEL):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        hidden_layers,
        time,
        skip,
        omega,
        sigma,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time = time
        self.skip = skip
        self.hidden_layers = hidden_layers

        self.nonlin = GaborLayer

        self.net = nn.ModuleList()
        self.net.append(self.nonlin(input_dim, hidden_dim, omega=omega, sigma=sigma))

        for i in range(hidden_layers):
            if skip and i == ceil(hidden_layers / 2):
                self.net.append(
                    self.nonlin(
                        hidden_dim + input_dim, hidden_dim, omega=omega, sigma=sigma
                    )
                )
            else:
                self.net.append(
                    self.nonlin(hidden_dim, hidden_dim, omega=omega, sigma=sigma)
                )

        final_linear = nn.Linear(hidden_dim, output_dim)

        self.net.append(final_linear)

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers / 2) + 1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x

