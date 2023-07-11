import torch
from torch import nn


class Sine(nn.Module):
    def __init__(self, omega_0=30):
        # number 30 is chosen experimentally from original paper
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, input):
        # print('sine embedding input : ',input)
        # print('sine embedding omega : ',omega_0)
        return torch.sin(self.omega_0 * input)
