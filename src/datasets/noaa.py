import os
import glob
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import Dataset

from utils.change_coord_sys import to_spherical

class NOAA(Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset_type,
        spherical=False,
        time=True,
        in_memory=True,
        **kwargs,
    ):
        self.dataset_dir = dataset_dir
        self.spherical = spherical
        self.time = time
        self.in_memory = in_memory
        self.filenames = self.get_filenames(dataset_dir, dataset_type)
        self.npzs = [np.load(f) for f in self.filenames]
        self._points_path = os.path.join(dataset_dir, dataset_type+"_points.npy")
        self._data = None
        self._points = None

        if in_memory:
            print(f"Loading {dataset_type} dataset")
            self._data = [self.load_data(i) for i in tqdm(range(len(self)))]

    def load_data(self, index):
        data = {}

        data["inputs"] = self.get_inputs(index)
        data["target"] = self.get_target(index)
        return data

    def get_points(self, index):
        if self._points is None:
            self._points = np.load(self._points_path)
            self._points = torch.from_numpy(self._points).float()
            if self.spherical:
                self._points = to_spherical(self._points)
        return self._points
        
    def get_time(self, index):
        arr = self.npzs[index]["time"]
        return torch.from_numpy(arr).float()

    def get_inputs(self, index):
        arr = self.get_points(index)
        if self.time:
            time = self.get_time(index)
            arr = self.add_time(arr, time)
        return arr

    def get_target(self, index):
        arr = self.npzs[index]["target"]
        return torch.from_numpy(arr).float()

    def get_data(self, index):
        if self.in_memory:
            return self._data[index]
        else:
            return self.load_data(index)

    def __getitem__(self, index):
        return self.get_data(index)

    def __len__(self):
        return len(self.filenames)

    @property
    def target_dim(self):
        return self.get_data(0)["target"].shape[-1]

    @staticmethod
    def add_time(points, time):
        n_points = points.shape[-2]
        time = time.unsqueeze(0).repeat(n_points, 1)
        return torch.cat([points, time], dim=-1)

    @staticmethod
    def get_filenames(dataset_dir, dataset_type):
        npz_dir = os.path.join(dataset_dir, "npz_files")
        npz_filenames = glob.glob(os.path.join(npz_dir, dataset_type+r"_*.npz"))
        npz_filenames = sorted(npz_filenames, key=lambda s: s.split("/")[-1])
        return npz_filenames