import os
import glob
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import numpy as np
import netCDF4 as nc
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import to_cartesian

from pathlib import Path
from scipy.spatial import ConvexHull
import pymesh
from scipy import sparse as sp
import re


class Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset_type,
        output_dim,
        normalize,
        panorama_idx,
        n_fourier=5,
        n_nodes_in_sample=5000,
        **kwargs,
    ):
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.normalize = normalize
        self.panorama_idx = panorama_idx
        self.n_fourier = n_fourier
        self.n_nodes_in_sample = n_nodes_in_sample
        self.filename = self.get_filenames()
        self._data = self.load_data()

    def __len__(self):
        return len(self.points_idx)

    def get_filenames(self):
        filenames = sorted(glob.glob(os.path.join(self.dataset_dir, "*.nc")))
        return (
            filenames[self.panorama_idx] if "360" in self.dataset_dir else filenames[0]
        )

    def __getitem__(self, index):
        data_out = dict()
        data_out["inputs"] = self._data["fourier"]  # [n/3, n_fourier]
        data_out["target"] = self._data["target"]  # [n/3, 1]

        data_out["inputs"] = data_out["inputs"][index]  # [n_fourier]
        data_out["target"] = data_out["target"][index]  # [1]

        return data_out

    def load_data(self):
        cachefile = Path(self.dataset_dir) / "cached" / "cached_data.npz"
        data_out = dict()

        if cachefile.is_file():
            print("Loading cached data:", self.dataset_type)
            data = np.load(cachefile)
        else:
            print("There is no cached data:", self.dataset_type, "/ Generating data...")
            if "360" in self.dataset_dir:
                target = np.array(Image.open(self.filename))  # [512, 1024, 3]

                H, W = target.shape[:2]  # H : 512, W : 1024

                lat = np.linspace(-90, 90, H)  # 512
                lon = np.linspace(-180, 180, W)  # 1024

            elif "era5" in self.dataset_dir:
                print(self.filename)
                with nc.Dataset(self.filename, "r") as f:
                    for variable in f.variables:
                        if variable == "latitude":
                            lat = f.variables[variable][:]
                        elif variable == "longitude":
                            lon = f.variables[variable][:]
                        elif variable == "z":
                            # Fixed
                            target = (
                                torch.tensor(f.variables[variable][0].data)
                                .unsqueeze(2)
                                .numpy()
                            )

            else:
                data = np.load(self.filename)

                lat = data["latitude"]
                lon = data["longitude"]
                target = data["target"]

            lats, lons = np.meshgrid(lat, lon)
            lats, lons = lats.T.data, lons.T.data  # (lat, lon), (lat, lon)
            xyz = sphere_to_cartesian(lats, lons)  # (lat, lon, 3)

            print("Triangulation")
            hull = ConvexHull(xyz.reshape(-1, 3))
            mesh = pymesh.form_mesh(hull.points, hull.simplices)
            points, adj = mesh_to_graph(mesh)

            print(f"Computing embeddings, size=({adj.shape})")
            u = get_fourier(adj)
            if self.normalize:
                target = (target - target.min()) / (target.max() - target.min())

            np.savez(
                cachefile,
                points=points,
                fourier=u,
                target=target.reshape(-1, 1),
                faces=mesh.faces,
            )
            data = np.load(cachefile)

        if self.dataset_type == "all":
            start, step = 0, 1
        elif self.dataset_type == "train":
            start, step = 0, 3
        elif self.dataset_type == "valid":
            start, step = 1, 3
        else:
            start, step = 2, 3

        self.n_points = data["target"].shape[0]
        self.points_idx = np.arange(start, self.n_points, step)

        data_out["fourier"] = torch.from_numpy(data["fourier"]).float()  # [n, 100]
        data_out["fourier"] = data_out["fourier"][
            self.points_idx, : self.n_fourier
        ]  # [n/3, 34]
        data_out["target"] = torch.from_numpy(data["target"]).float()  # [n, 1]
        data_out["target"] = data_out["target"][self.points_idx]  # [n/3, 1]
        return data_out

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


def sphere_to_cartesian(lat, lon):
    """
    Converts latitude and longitude coordinates in degrees, to x, y, z cartesian
    coordinates.
    """
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return np.stack([x, y, z], axis=-1)


def edges_to_adj(edges, n):
    a = sp.csr_matrix(
        (np.ones(edges.shape[:1]), (edges[:, 0], edges[:, 1])), shape=(n, n)
    )
    a = a + a.T
    a.data[:] = 1.0

    return a


def mesh_to_graph(mesh):
    points, edges = pymesh.mesh_to_graph(mesh)
    n = points.shape[0]
    adj = edges_to_adj(edges, n)

    return points, adj


def degree_matrix(A):
    degrees = np.array(A.sum(1)).flatten()
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def laplacian(A):
    return degree_matrix(A) - A


def get_fourier(adj, k=100):
    l = laplacian(adj)
    _, u = sp.linalg.eigsh(l, k=k, which="SM")
    n = l.shape[0]
    u *= np.sqrt(n)

    return u


if __name__ == "__main__":
    dd = Dataset(
        dataset_dir="../../dataset/spatial/era5_geopotential_100",
        dataset_type="train",
        output_dim=1,
        normalize="store_true",
        panorama_idx=1,
    )
    print(dd)
