import os
import glob

import torch
import numpy as np
import netCDF4 as nc
from torch.utils.data import Dataset

from utils.utils import to_cartesian, StandardScalerTorch, MinMaxScalerTorch


class Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset_type,
        output_dim,
        normalize,  # Choosing whether to normalize the time input
        zscore_normalize=False,  # Choosing whether to normalize the target
        data_year=None,  # Choosing which number of years to use
        time_resolution=1,  # Choosing the time resolution (1~8760, 1 for hour,24 for day, 168 for week)
        ratio=1,
        time_len=24,
        **kwargs
    ):
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.normalize = normalize
        self.zscore_normalize = zscore_normalize
        self.time_resolution = time_resolution
        self.ratio = ratio
        self.time_len = time_len
        self.scaler = None
        if self.normalize:
            self.scaler = MinMaxScalerTorch()
        elif self.zscore_normalize:
            self.scaler = StandardScalerTorch()
        self.filenames = self.get_filenames(data_year)
        # self._data = [self.load_data(filename) for filename in self.filenames]
        self._data = self.load_data(self.filenames)

    def __len__(self):
        return self._data["inputs"].size(0)

    def get_filenames(self, data_year=None):
        filenames = glob.glob(os.path.join(self.dataset_dir, "*.nc"))
        if data_year is not None:
            filenames = [filename for filename in filenames if data_year in filename]
        return sorted(filenames)[0]

    def __getitem__(self, index):
        data_out = dict()

        data = self._data
        data_out["inputs"] = data["inputs"][index]  # [4]
        data_out["target"] = data["target"][index]  #  [1]
        data_out["target_shape"] = data["target_shape"]  # [3]
        data_out["mean_lat_weight"] = data["mean_lat_weight"]  # [1]
        return data_out

    def load_data(self, filename):
        data_out = dict()

        with nc.Dataset(filename, "r") as f:
            for variable in f.variables:
                if variable == "latitude":
                    lat = f.variables[variable][:]
                elif variable == "longitude":
                    lon = f.variables[variable][:]
                elif variable == "time":
                    time = f.variables[variable][:]
                else:
                    target = f.variables[variable][:]

        """Convert lat, lon from degree to radian"""
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

        """Time resolution sampling, 1 for hour, 24 for day, 168 for week"""
        time_resolution_index = np.arange(0, len(time), self.time_resolution)
        time_resolution_index = time_resolution_index[: self.time_len * 2]
        time = time[time_resolution_index]
        target = target[time_resolution_index]

        """Time normalization: max_dim [8760], normalize to [0,1]"""
        time = (time - time.min()) / (time.max() - time.min())

        """Normalization of target"""  # scaler the data after sampling
        if self.zscore_normalize or self.normalize:
            self.scaler.fit(target.reshape(-1, 1))
            target = self.scaler.transform(target)  # tranformed into torch

        if self.dataset_type == "train":
            time_extra_len = np.arange(0, self.time_len, 1)
        elif self.dataset_type == "test":
            time_extra_len = np.arange(0, int(self.time_len * self.ratio), 1)
            time_extra_len += self.time_len

        time = time[time_extra_len]  # make 30 datasets
        target = target[time_extra_len]

        """Numpy to torch"""
        lat = torch.from_numpy(lat).float()
        lon = torch.from_numpy(lon).float()
        time = torch.from_numpy(time).float()

        """latitudinal weight for latitudinal weighting"""
        mean_lat_weight = torch.abs(torch.cos(lat)).mean().float()
        target_shape = target.shape

        """Making meshgrid and flatten"""
        time, lat, lon = torch.meshgrid(time, lat, lon)

        lat = lat.flatten()
        lon = lon.flatten()
        time = time.flatten()
        target = target.reshape(-1, self.output_dim)

        """concatenate lat, lon, time"""
        inputs = torch.stack([lat, lon, time], dim=-1)
        inputs = torch.cat([inputs[..., :2], inputs[..., 2:]], dim=-1)

        data_out["inputs"] = inputs
        data_out["target"] = target
        data_out["target_shape"] = target_shape
        data_out["mean_lat_weight"] = mean_lat_weight
        return data_out
