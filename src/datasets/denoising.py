import os
import glob

import torch
import numpy as np
import netCDF4 as nc
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import to_cartesian, add_noise

TAU = 3e1  # Photon noise
NOISE_SNR = 2


class Dataset(Dataset):
    def __init__(
        self, dataset_dir, dataset_type, output_dim, normalize, panorama_idx, **kwargs
    ):
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.normalize = normalize
        self.panorama_idx = panorama_idx
        self.filename = self.get_filenames()
        self._data = self.load_data()

    def __len__(self):
        return self._data["inputs"].size(0)

    def get_filenames(self):
        filenames = glob.glob(os.path.join(self.dataset_dir, "*"))
        return filenames[self.panorama_idx]

    def __getitem__(self, index):
        data_out = dict()

        data_out["inputs"] = self._data["inputs"][index]
        data_out["target"] = self._data["target"][index]
        data_out["g_target"] = self._data["g_target"][index]
        data_out["target_shape"] = self._data["target_shape"]
        data_out["mean_lat_weight"] = self._data["mean_lat_weight"]
        data_out["height"] = self._data["height"]
        data_out["width"] = self._data["width"]
        return data_out

    def load_data(self):
        data_out = dict()

        img = np.array(Image.open(self.filename))  # [1024, 2048, 3]

        target_shape = img.shape
        H, W = img.shape[:2]  # H : 1024, W : 2048

        if self.normalize:
            img = (img - img.min()) / (img.max() - img.min())
        noisy_img = add_noise(img, NOISE_SNR, TAU)
        img = torch.from_numpy(img)
        noisy_img = torch.from_numpy(noisy_img)

        lat = np.linspace(-90, 90, H)  # 1024 # 0, 180시도해보기
        lon = np.linspace(-180, 180, W)  # 2048

        lat = torch.from_numpy(np.deg2rad(lat)).float()
        lon = torch.from_numpy(np.deg2rad(lon)).float()

        mean_lat_weight = torch.cos(lat).mean()

        lat, lon = torch.meshgrid(lat, lon)  # [2048, 1024]for each
        lat = lat.flatten()
        lon = lon.flatten()
        # coords = np.hstack((lat.reshape(-1,1), lon.reshape(-1,1)))  # [2097152, 2]
        inputs = torch.stack([lat, lon], dim=-1)
        inputs = to_cartesian(inputs)  # [2097152, 3]

        # img = np.tensor(img).reshape(H*W,3)# [None,...] # [2097152, 3]
        # noisy_img = np.tensor(noisy_img).reshape(H*W,3)# [None,...] # [2097152, 3]
        img = img.reshape(-1, self.output_dim)
        noisy_img = noisy_img.reshape(-1, self.output_dim)

        data_out["inputs"] = inputs  # input coordinate [2097152,3]
        data_out["target"] = noisy_img  # noisy image pixel value [2097152,3]
        data_out["g_target"] = img  # ground truth image pixel value [2097152,3]
        data_out["target_shape"] = target_shape  # 2097152,3
        data_out["mean_lat_weight"] = mean_lat_weight  # 0.6360
        data_out["height"] = H
        data_out["width"] = W
        return data_out
