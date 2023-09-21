import numpy as np
import os

def gaussian_peak(lat_m, lon_m, peak_lat, peak_lon, height, width):
    lat_m, lon_m = np.radians(lat_m), np.radians(lon_m)
    peak_lat, peak_lon = np.radians(peak_lat), np.radians(peak_lon)
    
    delta_sigma = np.arccos(np.sin(lat_m) * np.sin(peak_lat) + np.cos(lat_m) * np.cos(peak_lat) * np.cos(lon_m - peak_lon))
    peak = height * np.exp(-((delta_sigma) ** 2) / (2 * (width ** 2)))
    return peak


def main(res, custom_fn, output_dir, peaks):
    lat_deg = np.linspace(-90, 90, int(180/res))
    lon_deg = np.linspace(0, 360, int(360/res))
    
    lon_m, lat_m = np.meshgrid(lon_deg, lat_deg)
    
    R = np.zeros_like(lon_m, dtype=float)
    
    for peak in peaks:
        peak_lat, peak_lon, height, width = peak
        R += gaussian_peak(lat_m, lon_m, peak_lat, peak_lon, height, width)
    
    data = {
        'latitude': lat_deg,
        'longitude': lon_deg,
        'target': R
    }

    np.savez(f'{output_dir}/{custom_fn}', **data)

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description='Generate and save data for Peaks on Sphere.')
#     parser.add_argument('--res', type=float, default=1, help='Resolution for the latitude and longitude grid in degrees.')
#     parser.add_argument('--custom_fn', type=str, default='data.npz', help='Filename to save the data.')
#     parser.add_argument('--output_dir', type=str, default='dataset/spatial/gaussian_peak', help='Directory to save the data.')
#     parser.add_argument('--peaks', type=lambda s: [tuple(map(float, peak.split(','))) for peak in s.split(';')],
#                         default="30,45,10,5", help='List of peaks as (lat, lon, height, width); another peak...')
    
#     args = parser.parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
#     main(args.res, args.custom_fn, args.output_dir, args.peaks)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate and save data for Peaks on Sphere.')
    parser.add_argument('--res', type=float, default=1, help='Resolution for the latitude and longitude grid in degrees.')
    parser.add_argument('--custom_fn', type=str, default='data.npz', help='Filename to save the data.')
    parser.add_argument('--output_dir', type=str, default='dataset/spatial/gaussian_peak', help='Directory to save the data.')
    parser.add_argument('--peaks', type=lambda s: [tuple(map(float, peak.split(','))) for peak in s.split(';')],
                        default=[(30,45,80,2)], help='List of peaks as (lat, lon, height, width); another peak...')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args.res, args.custom_fn, args.output_dir, args.peaks)
