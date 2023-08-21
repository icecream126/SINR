import os
import glob

import torch
import numpy as np
import netCDF4 as nc
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(
            self,
            dataset_dir,
            dataset_type,
            output_dim,
            normalize,
            **kwargs
        ):
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.normalize = normalize
        self.filenames = self.get_filenames()
        self._data = [self.load_data(filename) for filename in self.filenames]

    def __len__(self):
        return len(self.filenames)

    def get_filenames(self):
        filenames = glob.glob(os.path.join(self.dataset_dir, "*"))
        return sorted(filenames)

    def __getitem__(self, index):
        data_out = dict()

        data = self._data[index]
        data_out['inputs'] = data['inputs']
        data_out['target'] = data['target']
        data_out['mean_lat_weight'] = data['mean_lat_weight']
        return data_out

    def load_data(self, filename):
        data_out = dict()
        
        with nc.Dataset(filename, 'r') as f:
            for variable in f.variables:
                if variable == 'latitude':
                    lat = f.variables[variable][:]
                elif variable == 'longitude':
                    lon = f.variables[variable][:]
                elif variable == 'time':
                    time = f.variables[variable][:]
                elif variable in ['z', 't']:
                    target = f.variables[variable][:]
                        
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        time = (time - time.min()) / (time.max() - time.min())
        if self.normalize:
            target = (target - target.min()) / (target.max() - target.min())
        mean_lat_weight = np.cos(lat).mean()
        
        if self.dataset_type == 'train':
            start = 0 
        elif self.dataset_type == 'valid':
            start = 1
        else:
            start = 2

        lat_idx = np.arange(start, len(lat), 3)
        lon_idx = np.arange(start, len(lon), 3)
        time_idx = np.arange(start, len(time), 3)

        lat = torch.from_numpy(lat[lat_idx]).float()
        lon = torch.from_numpy(lon[lon_idx]).float()
        time = torch.from_numpy(time[time_idx]).float()
        target = torch.from_numpy(target[time_idx][:, lat_idx][:, :, lon_idx]).float()
        mean_lat_weight = torch.tensor(mean_lat_weight).float()

        time, lat, lon = torch.meshgrid(time, lat, lon)

        lat = lat.flatten()
        lon = lon.flatten()
        time = time.flatten()
        target = target.reshape(-1, self.output_dim)

        inputs = torch.stack([lat, lon, time], dim=-1)

        data_out['inputs'] = inputs
        data_out['target'] = target
        data_out['mean_lat_weight'] = mean_lat_weight
        return data_out