from argparse import ArgumentParser

import os
import glob
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import Dataset


class SwaveletDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        n_nodes_in_sample=10000,
        time=True,
        in_memory=True,
        dataset_type = 'train',
        **kwargs,
    ):
        self.dataset_dir = dataset_dir
        self.n_nodes_in_sample = n_nodes_in_sample
        self.time = time
        self._points = None
        self.in_memory = in_memory
        self._points_path = os.path.join(dataset_dir, dataset_type+"_spherical_points.npy")
        self.filenames = self.get_filenames(dataset_dir, dataset_type)
        self.npzs = [np.load(f) for f in self.filenames]
        self._data = None

        if in_memory:
            print("Loading dataset")
            self._data = [self.load_data(i) for i in tqdm(range(len(self)))]

    def load_data(self, index):
        data = {}

        data["inputs"] = self.get_inputs(index)
        data["target"] = self.get_target(index)

        return data

    def get_points(self, index):
        if os.path.exists(self._points_path):
            if self._points is None:
                self._points = np.load(self._points_path)
                self._points = torch.from_numpy(self._points).float()
            return self._points
        else:
            arr = self.npzs[index]["points"]
            return torch.from_numpy(arr).float()
        
    def get_time(self, index):
        arr = self.npzs[index]["time"]
        return torch.from_numpy(arr).float()

    @staticmethod
    def add_time(points, time):
        n_points = points.shape[-2]
        time = time.unsqueeze(0).repeat(n_points, 1)
        return torch.cat([points, time], dim=-1)

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
        data = self.get_data(index)
        data_out = dict()

        n_points = data["inputs"].shape[0]
        points_idx = self.get_subsampling_idx(n_points, self.n_nodes_in_sample)
        data_out["inputs"] = data["inputs"][points_idx]
        data_out["target"] = data["target"][points_idx]
        data_out["index"] = index

        return data_out

    def __len__(self):
        return len(self.filenames)

    @property
    def target_dim(self):
        return self.get_data(0)["target"].shape[-1]

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--dataset_dir", type=str)
        parser.add_argument("--n_nodes_in_sample", type=int, default=10000)
        parser.add_argument("--time", type=bool, default=True)
        parser.add_argument("--in_memory", type=bool, default=True)

        return parser

    @staticmethod
    def get_filenames(dataset_dir, dataset_type):
        npz_dir = os.path.join(dataset_dir, "npz_files")
        npz_filenames = glob.glob(os.path.join(npz_dir, dataset_type+r"_*.npz"))
        npz_filenames = sorted(npz_filenames, key=lambda s: s.split("/")[-1])

        return npz_filenames

    @staticmethod
    def get_subsampling_idx(n_points, to_keep):
        if n_points >= to_keep:
            idx = torch.randperm(n_points)[:to_keep]
        else:
            # Sample some indices more than once
            idx = (
                torch.randperm(n_points * int(np.ceil(to_keep / n_points)))[:to_keep]
                % n_points
            )

        return idx
