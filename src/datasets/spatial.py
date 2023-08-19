import torch
import numpy as np
from math import pi
import netCDF4 as nc
from PIL import Image
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(
            self,
            dataset_path,
            dataset_type,
            output_dim,
            **kwargs
        ):
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self._data = self.load_data()

    def __getitem__(self, index):
        data_out = dict()

        data_out['inputs'] = self._data['inputs'][index]
        data_out['target'] = self._data['target'][index]
        return data_out
    
    def __len__(self):
        return self._data['inputs'].size(0)
    
    @staticmethod
    def deg_to_rad(degrees):
        return pi * degrees / 180.

    def load_data(self):
        data_out = dict()

        if 'sun360' in self.dataset_path:
            target = np.array(Image.open(self.dataset_path))

            H, W = target.shape[:2]

            lat = np.linspace(-90, 90, H)
            lon = np.linspace(-180, 180, W)
        
        elif 'era5' in self.dataset_path:
            with nc.Dataset(self.dataset_path, 'r') as f:
                for variable in f.variables:
                    if variable == 'latitude':
                        lat = f.variables[variable][:]
                    elif variable == 'longitude':
                        lon = f.variables[variable][:]
                    elif variable in ['z', 't']:
                        target = f.variables[variable][0]
        
        else:
            data = np.load(self.dataset_path)

            lat = data['latitude']
            lon = data['longitude']
            target = data['target']

        lat = self.deg_to_rad(lat)
        lon = self.deg_to_rad(lon)
        target = (target - target.min()) / (target.max() - target.min())

        if self.dataset_type == 'train':
            start = 0 
        elif self.dataset_type == 'valid':
            start = 1
        else:
            start = 2

        lat_idx = np.arange(start, len(lat), 3)
        lon_idx = np.arange(start, len(lon), 3)

        lat = torch.from_numpy(lat[lat_idx]).float()
        lon = torch.from_numpy(lon[lon_idx]).float()
        target = torch.from_numpy(target[lat_idx][:, lon_idx]).float()

        lat, lon = torch.meshgrid(lat, lon)

        lat = lat.flatten()
        lon = lon.flatten()
        target = target.reshape(-1, self.output_dim)

        inputs = torch.stack([lat, lon], dim=-1)

        data_out['inputs'] = inputs
        data_out['target'] = target
        return data_out