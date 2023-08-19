import argparse

import numpy as np

def main(res, radius, custom_fn, output_dir):
    data = {}

    lat = np.arange(-90, 90, res)
    lon = np.arange(-180, 180, res)

    data['latitude'] = lat
    data['longitude'] = lon

    lon, lat = np.meshgrid(lon, lat)
    target = np.logical_and(lat>-radius, lat<radius).astype(np.float32)

    data['target'] = target

    np.savez(f'{output_dir}/{custom_fn}', **data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=float, default=0.25)
    parser.add_argument('--radius', type=float, default=45.)
    parser.add_argument('--custom_fn', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    main(args.res, args.radius, args.custom_fn, args.output_dir)