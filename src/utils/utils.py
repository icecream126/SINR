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

def add_noise(img, noise_snr, tau):
    # code from https://github.com/vishwa91/wire/blob/main/modules/utils.py#L85
    img_meas = np.copy(img)
    
    noise = np.random.randn(img_meas.size).reshape(img_meas.shape)*noise_snr
    
    if tau!= float('Inf'):
        img_meas = img_meas*tau
        
        img_meas[img_meas>0] = np.random.poisson(img_meas[img_meas>0])
        img_meas[img_meas<=0] = -np.random.poisson(-img_meas[img_meas<=0])
        
        img_meas = (img_meas + noise) / tau
    
    else:
        img_meas = img_meas + noise
    
    return img_meas