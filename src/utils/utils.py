import torch

def to_cartesian(points):
    theta, phi = points[..., 0], points[..., 1]

    x = torch.cos(theta) * torch.cos(phi)
    y = torch.cos(theta) * torch.sin(phi)
    z = torch.sin(theta)
    return torch.stack([x, y, z], dim=-1)

def mse2psnr(mse):
    """Computes PSNR from MSE, assuming the MSE was calculated between signals
    lying in [0, 1].

    Args:
        mse (torch.Tensor or float):
    """
    return -10.0 * torch.log10(mse)