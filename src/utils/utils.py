import torch
from math import pi

def to_cartesian(points):
    theta, phi = points[..., 0], points[..., 1]

    x = torch.cos(theta) * torch.cos(phi)
    y = torch.cos(theta) * torch.sin(phi)
    z = torch.sin(theta)
    return torch.stack([x, y, z], dim=-1)

def mse2psnr(mse):
    return -10.0 * torch.log10(mse)

def deg_to_rad(degrees):
    return pi * degrees / 180.