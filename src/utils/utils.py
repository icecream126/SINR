import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from numbers import Integral
import math

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x
    
    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin)/(xmax - xmin)

    return xnormalized

def measure(x, noise_snr=40, tau=100):
    ''' Realistic sensor measurement with readout and photon noise

        Inputs:
            noise_snr: Readout noise in electron count
            tau: Integration time. Poisson noise is created for x*tau.
                (Default is 100)

        Outputs:
            x_meas: x with added noise
    '''
    x_meas = np.copy(x)

    noise = np.random.randn(x_meas.size).reshape(x_meas.shape)*noise_snr

    # First add photon noise, provided it is not infinity
    if tau != float('Inf'):
        x_meas = x_meas*tau

        x_meas[x > 0] = np.random.poisson(x_meas[x > 0])
        x_meas[x <= 0] = -np.random.poisson(-x_meas[x <= 0])

        x_meas = (x_meas + noise)/tau

    else:
        x_meas = x_meas + noise

    return x_meas


# def mse2psnr(mse):
#     return -10.0 * np.log10(mse)


# def image_psnr(gt_image, noisy_image, weights):
#     gt_image = (255*gt_image).astype(np.uint8)
#     noisy_image = (255*noisy_image).astype(np.uint8)
    
#     error = np.sum((gt_image - noisy_image) ** 2, axis=-1).flatten()
#     mse = np.mean(weights * error)
#     print('mse : ',mse)
#     psnr = mse2psnr(mse)
#     return psnr

def psnr(x, xhat, weights):
    ''' Compute Peak Signal to Noise Ratio in dB

        Inputs:
            x: Ground truth signal
            xhat: Reconstructed signal

        Outputs:
            snrval: PSNR in dB
    '''
    err = x - xhat
    w_denom = np.mean(weights * pow(err, 2))
    w_snrval = 10*np.log10(np.max(x)/w_denom)
    return w_snrval


def crop(ar, crop_width, copy=False, order="K"):
    """Crop array `ar` by `crop_width` along each dimension.

    Parameters
    ----------
    ar : array-like of rank N
        Input array.
    crop_width : {sequence, int}
        Number of values to remove from the edges of each axis.
        ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies
        unique crop widths at the start and end of each axis.
        ``((before, after),) or (before, after)`` specifies
        a fixed start and end crop for every axis.
        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
        before = after = ``n`` for all axes.
    copy : bool, optional
        If `True`, ensure the returned array is a contiguous copy. Normally,
        a crop operation will return a discontiguous view of the underlying
        input array.
    order : {'C', 'F', 'A', 'K'}, optional
        If ``copy==True``, control the memory layout of the copy. See
        ``np.copy``.

    Returns
    -------
    cropped : array
        The cropped array. If ``copy=False`` (default), this is a sliced
        view of the input array.
    """
    ar = np.array(ar, copy=False)

    if isinstance(crop_width, Integral):
        crops = [[crop_width, crop_width]] * ar.ndim
    elif isinstance(crop_width[0], Integral):
        if len(crop_width) == 1:
            crops = [[crop_width[0], crop_width[0]]] * ar.ndim
        elif len(crop_width) == 2:
            crops = [crop_width] * ar.ndim
        else:
            raise ValueError(
                f"crop_width has an invalid length: {len(crop_width)}\n"
                f"crop_width should be a sequence of N pairs, "
                f"a single pair, or a single integer"
            )
    elif len(crop_width) == 1:
        crops = [crop_width[0]] * ar.ndim
    elif len(crop_width) == ar.ndim:
        crops = crop_width
    else:
        raise ValueError(
            f"crop_width has an invalid length: {len(crop_width)}\n"
            f"crop_width should be a sequence of N pairs, "
            f"a single pair, or a single integer"
        )

    slices = tuple(slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops))
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped


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
    theta = theta - math.pi / 2
    return torch.stack([theta, phi], dim=-1)


def deg_to_rad(degrees):
    return np.pi * degrees / 180.0


def add_noise(img, noise_snr, tau):
    # code from https://github.com/vishwa91/wire/blob/main/modules/utils.py#L85
    img_meas = np.copy(img)

    noise = np.random.randn(img_meas.size).reshape(img_meas.shape) * noise_snr

    if tau != float("Inf"):
        img_meas = img_meas * tau

        img_meas[img_meas > 0] = np.random.poisson(img_meas[img_meas > 0])
        img_meas[img_meas <= 0] = -np.random.poisson(-img_meas[img_meas <= 0])

        img_meas = (img_meas + noise) / tau

    else:
        img_meas = img_meas + noise

    return img_meas


def calculate_ssim(model, dataset, output_dim):
    data = dataset[:]

    inputs, target = data["inputs"], data["target"]
    inputs = to_cartesian(inputs)
    mean_lat_weight = data["mean_lat_weight"]
    target_shape = data["target_shape"]
    H, W = target_shape[:2]

    pred = model(inputs)

    weights = torch.cos(inputs[..., :1])
    weights = weights / mean_lat_weight
    weights = np.array(weights.reshape(H, W, -1))

    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    pred = pred.reshape(H, W, output_dim)
    target = target.reshape(H, W, output_dim)

    win_size = 7
    ssim, diff = ssim_func(target, pred, channel_axis=2, full=True, win_size=win_size)

    pad = (win_size - 1) // 2
    diff = diff * weights

    ssim = 0
    for i in range(output_dim):
        ssim += crop(diff[..., i], pad).mean()
    ssim = ssim / 3

    return ssim
