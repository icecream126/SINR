# Code from : https://github.com/EmilienDupont/coinpp/blob/main/coinpp/losses.py
import numpy as np
def mse2psnr(mse):
    """Computes PSNR from MSE, assuming the MSE was calculated between signals
    lying in [0, 1].

    Args:
        mse (torch.Tensor or float):
    """
    return -10.0 * np.log10(mse)