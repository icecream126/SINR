import torch
from torch import nn
from math import pi, ceil

from .model import MODEL
from .relu import ReLULayer

from utils.spherical_harmonics import components_from_spherical_harmonics


class SphericalHarmonicsLayer(nn.Module):
    def __init__(
        self,
        output_dim,
        levels,
        time,
        omega,
        **kwargs,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.levels = levels
        self.time = time
        self.omega = omega

        self.dilate = nn.Parameter(torch.empty(1, output_dim))
        nn.init.normal_(self.dilate)

        self.u = nn.Parameter(torch.empty(output_dim))
        self.v = nn.Parameter(torch.empty(output_dim))
        self.w = nn.Parameter(torch.empty(output_dim))
        nn.init.uniform_(self.u)
        nn.init.uniform_(self.v)
        nn.init.uniform_(self.w)

        if time:
            self.linear = nn.Linear(1, output_dim * levels**2)

    def forward(self, input):
        zeros = torch.zeros(self.output_dim, device=self.u.device)
        ones = torch.ones(self.output_dim, device=self.u.device)

        alpha = 2 * pi * self.u
        beta = torch.arccos(torch.clamp(2 * self.v - 1, -1 + 1e-6, 1 - 1e-6))
        gamma = 2 * pi * self.w

        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_alpha = torch.sin(alpha)
        sin_beta = torch.sin(beta)
        sin_gamma = torch.sin(gamma)

        Rz_alpha = torch.stack(
            [
                torch.stack([cos_alpha, -sin_alpha, zeros], 1),
                torch.stack([sin_alpha, cos_alpha, zeros], 1),
                torch.stack([zeros, zeros, ones], 1),
            ],
            1,
        )

        Rx_beta = torch.stack(
            [
                torch.stack([ones, zeros, zeros], 1),
                torch.stack([zeros, cos_beta, -sin_beta], 1),
                torch.stack([zeros, sin_beta, cos_beta], 1),
            ],
            1,
        )

        Rz_gamma = torch.stack(
            [
                torch.stack([cos_gamma, -sin_gamma, zeros], 1),
                torch.stack([sin_gamma, cos_gamma, zeros], 1),
                torch.stack([zeros, zeros, ones], 1),
            ],
            1,
        )

        R = torch.bmm(torch.bmm(Rz_gamma, Rx_beta), Rz_alpha)

        points = input[..., 0:3]
        points = torch.matmul(R, points.unsqueeze(-2).unsqueeze(-1))
        points = points.squeeze(-1)

        x, y, z = points[..., 0], points[..., 1], points[..., 2]

        a = torch.exp(self.dilate)
        A = a * a * (1 - z) + (1 + z)

        x, y, z = 2 * a * x / A, 2 * a * y / A, 2 * (1 + z) / A

        coord = torch.stack([x, y, z], dim=-1)

        out = components_from_spherical_harmonics(self.levels, coord).view(
            *input.shape[:-1], -1
        )

        if self.time:
            time = input[..., 3:]
            lin = self.linear(time)
            omega = self.omega * lin
            out = out * torch.sin(omega)
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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time = time
        self.skip = skip
        self.hidden_layers = hidden_layers

        self.first_nonlin = SphericalHarmonicsLayer

        first_hidden_dim = hidden_dim // levels**2

        self.net = nn.ModuleList()
        self.net.append(self.first_nonlin(first_hidden_dim, levels, time, omega))

        self.nonlin = ReLULayer

        for i in range(hidden_layers):
            if i == 0:
                self.net.append(self.nonlin(first_hidden_dim * levels**2, hidden_dim))
            elif skip and i == ceil(hidden_layers / 2):
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
