from math import pi, sqrt
from functools import reduce, wraps
from operator import mul
import torch
import math
import numpy as np
from scipy.special import lpmv

from functools import lru_cache

def cache(cache, key_fn):
    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache:
                return cache[key_name]
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner
    return cache_inner
# constants

#CACHE = {}

#def clear_spherical_harmonics_cache():
#    CACHE.clear()

#def lpmv_cache_key_fn(l, m, x):
#    return (l, m)

# spherical harmonics

#@lru_cache(maxsize = 1000)
def semifactorial(x):
    return reduce(mul, range(x, 1, -2), 1.)

#@lru_cache(maxsize = 1000)
def pochhammer(x, k):
    return reduce(mul, range(x + 1, x + k), float(x))

def negative_lpmv(l, m, y):
    if m < 0:
        y *= ((-1) ** m / pochhammer(l + m + 1, -2 * m))
    return y

#@cache(cache = CACHE, key_fn = lpmv_cache_key_fn)
def lpmv(l, m, x):
    """Associated Legendre function including Condon-Shortley phase.

    Args:
        m: int order 
        l: int degree
        x: float argument tensor
    Returns:
        tensor of x-shape
    """
    # Check memoized versions
    m_abs = abs(m)

    if m_abs > l:
        return None

    if l == 0:
        return torch.ones_like(x)
    
    # Check if on boundary else recurse solution down to boundary
    if m_abs == l:
        # Compute P_m^m
        y = (-1)**m_abs * semifactorial(2*m_abs-1)
        y *= torch.pow(1-x*x, m_abs/2)
        return negative_lpmv(l, m, y)

    # Recursively precompute lower degree harmonics
    #lpmv(l-1, m, x)

    # Compute P_{l}^m from recursion in P_{l-1}^m and P_{l-2}^m
    # Inplace speedup
    y = ((2*l-1) / (l-m_abs)) * x * lpmv(l-1, m_abs, x)

    if l - m_abs > 1:
        y -= ((l+m_abs-1)/(l-m_abs)) * lpmv(l-2, m_abs, x)
    
    if m < 0:
        y = self.negative_lpmv(l, m, y)
    return y

def get_spherical_harmonics_element(l, m, theta, phi):
    """Tesseral spherical harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        m: int for order, where -l <= m < l
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape theta
    """
    m_abs = abs(m)
    assert m_abs <= l, "absolute value of order m must be <= degree l"

    N = sqrt((2*l + 1) / (4 * pi))
    leg = lpmv(l, m_abs, torch.cos(theta))

    if m == 0:
        return N * leg

    if m > 0:
        Y = torch.cos(m * phi)
    else:
        Y = torch.sin(m_abs * phi)

    Y *= leg
    
    ####### IMPORTANT #######
    # somehow works better without normalization...
    #N *= sqrt(2. / pochhammer(l - m_abs + 1, 2 * m_abs))
    ####### IMPORTANT #######
    
    Y *= N
    return Y

def get_spherical_harmonics(l, theta, phi):
    """ Tesseral harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape [*theta.shape, 2*l+1]
    """
    return torch.cat([ get_spherical_harmonics_element(l, m, theta, phi) \
                         for m in range(-l, l+1) ],
                        dim = -1)


# def get_spherical_harmonics_by_np(max_order, theta, phi):
#     theta = theta.detach().cpu().numpy()
#     phi = phi.detach().cpu().numpy()

#     y = []
#     for l in range(max_order + 1):
#         for m in range(-l, l + 1):
#             Klm = math.sqrt((2*l+1) * math.factorial(l-abs(m)) / (4*math.pi * math.factorial(l+abs(m))))
            
#             if m > 0:
#                 Ylm = Klm * math.sqrt(2) * lpmv(m, l, np.cos(theta)) * np.cos(m * phi)
#             elif m == 0:
#                 Ylm = Klm * lpmv(0, l, np.cos(theta))
#             else:
#                 Ylm = Klm * math.sqrt(2) * lpmv(-m, l, np.cos(theta)) * np.sin(-m * phi)

#             y.append(torch.Tensor(Ylm).cuda())

#     return y

# def get_wrong_spherical_harmonics_by_np(max_order, theta, phi):
#     theta = theta.detach().cpu().numpy()
#     phi = phi.detach().cpu().numpy()

#     y = []
#     for l in range(max_order + 1):
#         for m in range(-l, l + 1):
#             Klm = math.sqrt((2*l+1) * math.factorial(l-m) / (4*math.pi * math.factorial(l+m)))
            
#             if m > 0:
#                 Ylm = Klm * math.sqrt(2) * lpmv(m, l, np.cos(theta)) * np.cos(m * phi)
#             elif m == 0:
#                 Ylm = Klm * lpmv(0, l, np.cos(theta))
#             else:
#                 Ylm = Klm * math.sqrt(2) * lpmv(-m, l, np.cos(theta)) * np.sin(-m * phi)

#             y.append(torch.Tensor(Ylm).cuda())

#     return y