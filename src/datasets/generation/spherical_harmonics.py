import argparse
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
import os

def main(l, m, res, custom_fn, output_dir):
    lat_deg = np.linspace(-90, 90, int(180/res))  # Latitude: -90 to 90 degrees
    lon_deg = np.linspace(0, 360, int(360/res))  # Longitude: 0 to 360 degrees
    
    
    lon_m, lat_m = np.meshgrid(lon_deg, lat_deg)
    
    # Convert lat, lon in degrees to THETA, PHI in radians for spherical harmonics
    THETA = (lat_m + 90) * np.pi / 180
    PHI = lon_m * np.pi / 180
    
    R = sp.sph_harm(m, l, PHI, THETA).real  # Spherical Harmonics for the meshgrid of PHI, THETA

    lat_rad, lon_rad = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    data = {
        'latitude': lat_rad,
        'longitude': lon_rad,
        'target': R
    }

    np.savez(f'{output_dir}/{custom_fn}', **data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize and save data for Spherical Harmonics.')
    parser.add_argument('--l', type=int, default=3, help='Degree of the spherical harmonics.')
    parser.add_argument('--m', type=int, default=3, help='Order of the spherical harmonics.')
    parser.add_argument('--res', type=float, default=1, help='Resolution for the latitude and longitude grid in degrees.')
    parser.add_argument('--custom_fn', type=str, default='data.npz', help='Filename to save the data and image.')
    parser.add_argument('--output_dir', type=str, default='dataset/spatial/spherical_harmonics', help='Directory to save the data and image.')

    args = parser.parse_args()
    args.output_dir = args.output_dir+'_'+str(args.l)+'_'+str(args.m)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args.l, args.m, args.res, args.custom_fn, args.output_dir)
