import os
import glob
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import numpy as np
import netCDF4 as nc
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import (
    to_cartesian,
    to_spherical,
    StandardScalerTorch,
    MinMaxScalerTorch,
)

from pathlib import Path
from scipy.spatial import ConvexHull

# import pymesh
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
        zscore_normalize=False,
        **kwargs,
    ):
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.normalize = normalize
        self.panorama_idx = panorama_idx
        self.n_fourier = n_fourier
        self.n_nodes_in_sample = n_nodes_in_sample
        self.zscore_normalize = zscore_normalize
        self.scaler = None
        if self.normalize:
            self.scaler = MinMaxScalerTorch()
        elif self.zscore_normalize:
            self.scaler = StandardScalerTorch()
        self.filename = self.get_filenames()
        self._data = self.load_data()

    def __len__(self):
        return self._data["fourier"].shape[0]

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
        data_out["mean_lat_weight"] = self._data["mean_lat_weight"]
        data_out["spherical"] = self._data["spherical"][index]
        data_out["target_shape"] = self._data["target_shape"]
        return data_out

    def load_data(self):
        cur_filename = os.path.basename(self.filename).split(".")[0]
        cachefile = (
            Path(self.dataset_dir)
            / "cached"
            / (cur_filename + "_" + self.dataset_type + ".npz")
        )
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
                            lat = f.variables[variable][::-1]
                        elif variable == "longitude":
                            lon = f.variables[variable][:]
                        elif variable == "z":
                            target = f.variables[variable][0][::-1]
                            # (lat, lon) , [0] implies time=0
            else:
                data = np.load(self.filename)

                lat = data["latitude"]
                lon = data["longitude"]
                target = data["target"]

            """Sampling index"""
            if self.dataset_type == "all":
                start, step = 0, 1
            elif self.dataset_type == "train":
                start, step = 0, 2
            elif self.dataset_type == "test":
                start, step = 1, 2
            else:
                start, step = 0, 2

            """Sampling data"""  # if train -> using sampled, test -> using all but data for sampled
            lat_sample = np.arange(start, lat.shape[0], step)
            lon_sample = np.arange(start, lon.shape[0], step)

            if self.dataset_type == "train":
                lats = lat[lat_sample]
                lons = lon[lon_sample]
                target = target[lat_sample][:, lon_sample]
            else:
                lats = lat
                lons = lon

            lats, lons = lats.astype(np.float64), lons.astype(np.float64)
            lats, lons = np.meshgrid(lats, lons)
            lats, lons = lats.T.data, lons.T.data  # (lat, lon), (lat, lon)
            xyz = sphere_to_cartesian(lats, lons)  # (lat, lon, 3)
            mean_lat_weight = torch.abs(torch.cos(torch.Tensor(lats))).mean()  # 0.6341

            print("Triangulation")
            hull = ConvexHull(xyz.reshape(-1, 3))
            mesh = pymesh.form_mesh(hull.points, hull.simplices)
            points, adj = mesh_to_graph(mesh)

            print(f"Computing embeddings, size=({adj.shape})")
            four = get_fourier(adj)

            if self.dataset_type == "test":
                four = four.reshape(lat.shape[0], lon.shape[0], -1)[lat_sample][
                    :, lon_sample
                ]
                four = four.reshape(-1, four.shape[-1])
                target = target.reshape(lat.shape[0], lon.shape[0], -1)[lat_sample][
                    :, lon_sample
                ]
            # index = np.arange(lat.shape[0]*lon.shape[0]).reshape(lat.shape[0], lon.shape[0])[lat_sample][:, lon_sample].reshape(-1)
            os.makedirs(Path(self.dataset_dir) / "cached", exist_ok=True)
            np.savez(
                cachefile,
                points=points,
                fourier=four,
                target_shape=target.shape,
                target=target.reshape(-1, 1),
                faces=mesh.faces,
                mean_lat_weight=mean_lat_weight,
            )
            data = np.load(cachefile)

        data_out["fourier"] = torch.from_numpy(data["fourier"]).float()[
            :, -self.n_fourier :
        ]  # [n, 100] -> [n, n_fourier]
        data_out["target"] = torch.from_numpy(data["target"]).float()  # [n, 1]]
        data_out["mean_lat_weight"] = data["mean_lat_weight"]  # 0.6341
        data_out["points"] = torch.from_numpy(data["points"]).float()  # [n, 3]
        data_out["spherical"] = cartesian_to_sphere(data_out["points"])  # [n, 2]
        data_out["target_shape"] = data["target_shape"]  # [171, 342, 3]

        if self.zscore_normalize or self.normalize:
            self.scaler.fit(data_out["target"])
            data_out["target"] = self.scaler.transform(data_out["target"])

        return data_out


def sphere_to_cartesian(lat, lon):
    """
    Converts latitude and longitude coordinates in degrees, to x, y, z cartesian
    coordinates.
    """
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return np.stack([x, y, z], axis=-1)


def cartesian_to_sphere(points):
    """
    Converts x, y, z cartesian coordinates to latitude and longitude coordinates
    N.B.: NetCDF uses a 0:360 range for longitude
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]

    lat = np.arcsin(z)
    lon = np.arctan2(y, x)

    return torch.stack([lat, lon], dim=-1)


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
    _, u = sp.linalg.eigs(l, k=k, which="SM")
    # _, u = sp.linalg.eigsh(l, k=k, which="SM")
    n = l.shape[0]
    u *= np.sqrt(n)

    return u


if __name__ == "__main__":
    dd = Dataset(
        dataset_dir="../../dataset/spatial/era5_geopotential_100",
        dataset_type="test",
        output_dim=1,
        normalize="store_true",
        panorama_idx=1,
    )
    print(dd)
