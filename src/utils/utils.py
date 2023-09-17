import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from numbers import Integral
import math
from torch import nn

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def image_psnr(gt_image, noisy_image, weights):
    error = (gt_image - noisy_image) ** 2
    mse = np.mean(weights * error)
    psnr = mse2psnr(mse)
    return psnr



def normalize(x, fullnormalize=False):
    """
    Normalize input to lie between 0, 1.

    Inputs:
        x: Input signal
        fullnormalize: If True, normalize such that minimum is 0 and
            maximum is 1. Else, normalize such that maximum is 1 alone.

    Outputs:
        xnormalized: Normalized x.
    """

    if x.sum() == 0:
        return x

    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin) / (xmax - xmin)

    return xnormalized


def measure(x, noise_snr=40, tau=100):
    """Realistic sensor measurement with readout and photon noise

    Inputs:
        noise_snr: Readout noise in electron count
        tau: Integration time. Poisson noise is created for x*tau.
            (Default is 100)

    Outputs:
        x_meas: x with added noise
    """
    x_meas = np.copy(x)

    noise = np.random.randn(x_meas.size).reshape(x_meas.shape) * noise_snr

    # First add photon noise, provided it is not infinity
    if tau != float("Inf"):
        x_meas = x_meas * tau

        x_meas[x > 0] = np.random.poisson(x_meas[x > 0])
        x_meas[x <= 0] = -np.random.poisson(-x_meas[x <= 0])

        x_meas = (x_meas + noise) / tau

    else:
        x_meas = x_meas + noise

    return x_meas


def mse2psnr(mse):
    return -10.0 * np.log10(mse)


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

    weights = torch.abs(torch.cos(inputs[..., :1]))
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


def geometric_initializer(layer, in_dim):
    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.00001)
    nn.init.constant_(layer.bias, -1)


def first_layer_sine_initializer(layer):
    with torch.no_grad():
        if hasattr(layer, "weight"):
            num_input = layer.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            layer.weight.uniform_(-1 / num_input, 1 / num_input)


def sine_initializer(layer):
    with torch.no_grad():
        if hasattr(layer, "weight"):
            num_input = layer.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            layer.weight.uniform_(
                -np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30
            )


class Sine(nn.Module):
    def __init__(self, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, input):
        return torch.sin(self.omega_0 * input)


EPSILON = 1e-5


class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        # X = X.clone().detach()
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        # X = X.clone().detach()
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(), stds=self.stds.clone().detach()
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )


class MinMaxScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, max=None, min=None):
        self.max = max
        self.min = min

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float)
        self.max = torch.max(X, dim=0)[0]
        self.min = torch.min(X, dim=0)[0]

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        # X = X.clone().detach()
        return (X - self.max) / (self.max - self.min + EPSILON)

    def inverse_transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        # X = X.clone().detach()
        return X * (self.max - self.min) + self.min

    def match_device(self, tensor):
        if self.max.device != tensor.device:
            self.max = self.max.to(tensor.device)
            self.min = self.min.to(tensor.device)

    def copy(self):
        return MinMaxScalerTorch(
            maxs=self.max.clone().detach(), min=self.min.clone().detach()
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max: {self.max.tolist()}, "
            f"min: {self.min.tolist()})"
        )
