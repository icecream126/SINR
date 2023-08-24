import os
import glob

import torch
import numpy as np
import netCDF4 as nc
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import to_cartesian

class Dataset(Dataset):
    def __init__(
            self,
            dataset_dir,
            dataset_type,
            output_dim,
            normalize,
            panorama_idx,
            **kwargs
        ):
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.normalize = normalize
        self.panorama_idx = panorama_idx
        self.filename = self.get_filenames()
        self._data = self.load_data()
        
    def __len__(self):
        return self._data['inputs'].size(0)

    def get_filenames(self):
        filenames = glob.glob(os.path.join(self.dataset_dir, "*"))
        return filenames[self.panorama_idx] if 'sun360' in self.dataset_dir else filenames[0]

    def __getitem__(self, index):
        data_out = dict()

        data_out['inputs'] = self._data['inputs'][index]
        data_out['target'] = self._data['target'][index]
        data_out['target_shape'] = self._data['target_shape']
        data_out['mean_lat_weight'] = self._data['mean_lat_weight']
        return data_out

    def load_data(self):
        data_out = dict()

        if 'sun360' in self.dataset_dir:
            target = np.array(Image.open(self.filename))

            H, W = target.shape[:2]

            lat = np.linspace(-90, 90, H)
            lon = np.linspace(-180, 180, W)
        
        elif 'era5' in self.dataset_dir:
            with nc.Dataset(self.filename, 'r') as f:
                for variable in f.variables:
                    if variable == 'latitude':
                        lat = f.variables[variable][:]
                    elif variable == 'longitude':
                        lon = f.variables[variable][:]
                    elif variable in ['z', 't']:
                        target = f.variables[variable][0]
        else:
            data = np.load(self.filename)

            lat = data['latitude']
            lon = data['longitude']
            target = data['target']

        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

        if self.normalize:
            target = (target - target.min()) / (target.max() - target.min())

        if self.dataset_type == 'all':
            start, step = 0, 1
        elif self.dataset_type == 'train':
            start, step = 0, 3 
        elif self.dataset_type == 'valid':
            start, step = 1, 3 
        else:
            start, step = 2, 3 

        lat_idx = np.arange(start, len(lat), step)
        lon_idx = np.arange(start, len(lon), step)

        lat = torch.from_numpy(lat[lat_idx]).float()
        lon = torch.from_numpy(lon[lon_idx]).float()
        target = torch.from_numpy(target[lat_idx][:, lon_idx]).float()

        mean_lat_weight = torch.cos(lat).mean()
        target_shape = target.shape

        lat, lon = torch.meshgrid(lat, lon)

        lat = lat.flatten()
        lon = lon.flatten()
        target = target.reshape(-1, self.output_dim)

        inputs = torch.stack([lat, lon], dim=-1)
        inputs = to_cartesian(inputs)

        data_out['inputs'] = inputs
        data_out['target'] = target
        data_out['target_shape'] = target_shape
        data_out['mean_lat_weight'] = mean_lat_weight
        return data_out