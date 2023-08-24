import torch
import numpy as np

def to_cartesian(points):
    theta, phi = points[..., 0], points[..., 1]

    x = torch.cos(theta) * torch.cos(phi)
    y = torch.cos(theta) * torch.sin(phi)
    z = torch.sin(theta)
    return torch.stack([x, y, z], dim=-1)

def to_spherical(points):
    x, y, z = points[..., 0], points[..., 1], points[..., 2]

    theta = torch.acos(z)
    phi = torch.atan2(y, x)
    return torch.stack([theta, phi], dim=-1)

def mse2psnr(mse):
    return -10.0 * np.log10(mse)

def deg_to_rad(degrees):
    return np.pi * degrees / 180.