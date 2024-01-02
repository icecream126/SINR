import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from numbers import Integral
import math
from torch import nn
import cv2
from math import sin, cos, sqrt, atan2, radians
import healpy as hp
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mhealpy import HealpixMap


# https://gist.github.com/zonca/680c68c3d60697eb0cb669cf1b41c324
def gethealpixmap(data, nside, theta, phi):
    '''
    data (np.array) : flattened target value 
    nside (int) : 2^k
    theta (np.array) : flattened latitude (0~pi)
    phi (np.array) : flattened longitude (0~2*pi)
    '''
    pixel_indices = hp.ang2pix(nside, theta, phi)
    
    # Single resolution map
    m = np.zeros(hp.nside2npix(nside))
    m[pixel_indices] = data
    hp.mollview(m, title="My HEALPix Map", cmap='viridis')
    plt.grid(True)  # Adds a grid to the map
    plt.savefig("healpix_map.png", dpi=300)  
    plt.close()
    
    # Multi resolution map    
    mm = HealpixMap(nside = nside)
    mm[pixel_indices] += data
    mm = mm.to_moc(max_value = max(mm))
    
    # TODO : Visualize mm
    
    return m, mm
    
    


# https://github.com/ai4cmb/NNhealpix/blob/master/nnhealpix/projections/__init__.py
def img2healpix_planar(img, nside, thetac, phic, delta_theta, delta_phi, rot=None):
    """Project a 2D image on healpix map

    Args:
        * img (array): image to project. It must have shape ``(#img,
          M, N)``
        * nside (int): ``NSIDE`` parameter for the output map.
        * thetac, phic (float): coordinates (in degrees) where to
          project the center of the image on the healpix map. They
          must follow the HEALPix angle convention:
            - ``0 <= thetac <= 180``, with 0 being the N and 180 the S Pole
            - ``0 <= phic <= 360``, with 0 being at the center of the
              map. It increases moving towards W
        * delta_theta, delta_phi (float): angular size of the projected image
        * rot: not implemented yet!

    Returns:
        The HEALPix map containing the projected image.
    """
    # img = img.unsqueeze(-1)
    imgf = np.flip(img, axis=2)
    imgf = np.array(imgf)

    data = imgf.reshape(img.shape[0], img.shape[1] * img.shape[2])
    xsize = img.shape[1]
    ysize = img.shape[2]
    theta_min = thetac - delta_theta / 2.0
    theta_max = thetac + delta_theta / 2.0
    phi_max = phic + delta_phi / 2.0
    phi_min = phic - delta_phi / 2.0
    theta_min = np.radians(theta_min)
    theta_max = np.radians(theta_max)
    phi_min = np.radians(phi_min)
    phi_max = np.radians(phi_max)
    img_theta_temp = np.linspace(theta_min, theta_max, ysize)
    img_phi_temp = np.linspace(phi_min, phi_max, xsize)
    ipix = np.arange(hp.nside2npix(nside))
    if rot == None:
        theta_r, phi_r = hp.pix2ang(nside, ipix)
    theta1 = theta_min
    theta2 = theta_max
    flg = np.where(theta_r < theta1, 0, 1)
    flg *= np.where(theta_r > theta2, 0, 1)
    if phi_min >= 0:
        phi1 = phi_min
        phi2 = phi_max
        flg *= np.where(phi_r < phi1, 0, 1)
        flg *= np.where(phi_r > phi2, 0, 1)
    else:
        phi1 = 2.0 * np.pi + phi_min
        phi2 = phi_max
        flg *= np.where((phi2 < phi_r) & (phi_r < phi1), 0, 1)
        img_phi_temp[img_phi_temp < 0] = 2 * np.pi + img_phi_temp[img_phi_temp < 0]
    img_phi, img_theta = np.meshgrid(img_phi_temp, img_theta_temp)
    img_phi = img_phi.flatten()
    img_theta = img_theta.flatten()
    ipix = np.compress(flg, ipix)
    pl_theta = np.compress(flg, theta_r)
    pl_phi = np.compress(flg, phi_r)
    points = np.zeros((len(img_theta), 2), "d")
    points[:, 0] = img_theta
    points[:, 1] = img_phi
    npix = hp.nside2npix(nside)
    hp_map = np.zeros((data.shape[0], npix), "d")
    for i in range(data.shape[0]):
        hp_map[i, ipix] = griddata(
            points, data[i, :], (pl_theta, pl_phi), method="nearest"
        )
    return hp_map

def calculate_parameter_size(model):
    total_size = 0
    for param in model.parameters():
        total_size += param.numel()  # numel() returns the total number of elements in the input tensor
    return total_size



def geodesic():
    # Approximate radius of earth in km
    R = 1.0

    lat1 = radians(52.2296756)
    lon1 = radians(21.0122287)
    lat2 = radians(52.406374)
    lon2 = radians(16.9251681)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c


# https://github.com/V-Sense/360SR/blob/master/ws_ssim.py (Candidate)
# https://github.com/Fanghua-Yu/OSRT/blob/master/odisr/metrics/odi_metric.py
def genERP(j, N):
    val = math.pi / N
    w = math.cos((j - (N / 2) + 0.5) * val)
    return w


def _ws_ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    equ = np.zeros((ssim_map.shape[0], ssim_map.shape[1]))

    for i in range(0, equ.shape[0]):
        for j in range(0, equ.shape[1]):
            equ[i, j] = genERP(i, equ.shape[0])

    return np.multiply(ssim_map, equ).mean() / equ.mean()


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
        return (X - self.min) / (self.max - self.min + EPSILON)

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
