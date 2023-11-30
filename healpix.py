'''
import healpy as hp
import numpy as np

def bilinear_interpolation(coords, healpix_map):
    # Get the nside from the HEALPix map
    nside = hp.get_nside(healpix_map)

    interpolated_values = []

    for ra, dec in coords:
        # Convert RA, DEC to HEALPix theta and phi
        theta = np.radians(90.0 - dec)  # Theta is 0 at the north pole and π at the south pole
        phi = np.radians(ra)  # Phi ranges from 0 to 2π

        # Find the pixels surrounding the point
        vec = hp.ang2vec(theta, phi)
        pixels = hp.get_all_neighbours(nside, theta, phi, lonlat=False)

        # Filter out bad pixels (which can be -1)
        good_pixels = pixels[pixels >= 0]
        if len(good_pixels) == 0:
            interpolated_values.append(np.nan)  # Return NaN if no good neighbors
            continue

        # Get values from the map
        values = healpix_map[good_pixels]

        # Compute weights for interpolation
        distances = np.sqrt(np.sum((hp.pix2vec(nside, good_pixels) - vec[:, None])**2, axis=0))
        weights = 1 / distances

        # Normalize weights
        weights /= np.sum(weights)

        # Interpolate
        interpolated_value = np.sum(values * weights)
        interpolated_values.append(interpolated_value)

    return interpolated_values

# Example usage
nside = 64  # Example Nside
healpix_map = np.random.rand(hp.nside2npix(nside))  # Example map

coords = [(120.0, -30.0), (130.0, -35.0)]  # Example list of coordinates
interpolated_values = bilinear_interpolation(coords, healpix_map)
print("Interpolated Values:", interpolated_values)
'''

import healpy as hp
import numpy as np

def bilinear_interpolation(coords, healpix_map):
    # Get the nside from the HEALPix map
    nside = hp.get_nside(healpix_map)

    interpolated_values = []

    for ra, dec in coords:
        # Convert RA, DEC to HEALPix theta and phi
        theta = np.radians(90.0 - dec)  # Theta is 0 at the north pole and π at the south pole
        phi = np.radians(ra)  # Phi ranges from 0 to 2π

        # Find the pixels surrounding the point
        vec = hp.ang2vec(theta, phi)
        pixels = hp.get_all_neighbours(nside, theta, phi, lonlat=False)

        # Filter out bad pixels (which can be -1)
        good_pixels = pixels[pixels >= 0]
        if len(good_pixels) == 0:
            interpolated_values.append(np.nan)  # Return NaN if no good neighbors
            continue

        # Get values from the map
        values = healpix_map[good_pixels]

        # Calculate great-circle distances to the neighbor pixels
        neighbor_vecs = hp.pix2vec(nside, good_pixels)
        distances = hp.rotator.angdist(vec, neighbor_vecs)

        # Compute weights for interpolation
        # Using inverse of great-circle distance
        weights = 1 / distances

        # Normalize weights
        weights /= np.sum(weights)

        # Interpolate
        interpolated_value = np.sum(values * weights)
        interpolated_values.append(interpolated_value)

    return interpolated_values

# Example usage
nside = 64  # Example Nside
healpix_map = np.random.rand(hp.nside2npix(nside))  # Example map

coords = [(120.0, -30.0), (130.0, -35.0)]  # Example list of coordinates
interpolated_values = bilinear_interpolation(coords, healpix_map)
print("Interpolated Values:", interpolated_values)
