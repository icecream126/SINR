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
from utils.utils import to_cartesian, StandardScalerTorch, MinMaxScalerTorch


class Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset_type,
        output_dim,
        normalize,
        n_fourier=5,  # Chosen number of fourier features (1~100, 34 for weather)
        n_nodes_in_sample=5000,
        zscore_normalize=False,  # Choosing whether to normalize the target
        data_year="2018",  # Choosing which number of years to use
        time_resolution=1,  # Choosing the time resolution (1~8760, 1 for hour,24 for day, 168 for week)
        **kwargs,
    ):
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.normalize = normalize
        self.n_fourier = n_fourier
        self.n_nodes_in_sample = n_nodes_in_sample
        self.zscore_normalize = zscore_normalize
        self.time_resolution = time_resolution
        self.filenames = self.get_filenames(data_year)
        self.scaler = None
        if self.normalize:
            self.scaler = MinMaxScalerTorch()
        elif self.zscore_normalize:
            self.scaler = StandardScalerTorch()

        self._data = self.load_data(self.filenames)

    def __len__(self):
        return self._data["target"].size(0)

    def get_filenames(self, data_year=None):
        filenames = glob.glob(os.path.join(self.dataset_dir, "*.nc"))
        if data_year is not None:
            filenames = [filename for filename in filenames if data_year in filename]
        return sorted(filenames)[0]

    def __getitem__(self, index):
        data_out = dict()
        data = self._data

        l1 = data["fourier"].size(0)
        # l2 = data["time"].size(0)

        data_out["inputs"] = data["fourier"][index % l1]  # [n_fourier]
        data_out["time"] = data["time"][index // l1]  # [1]
        data_out["target"] = data["target"][index]  # [1]

        # import pdb

        # pdb.set_trace()

        return data_out

    def load_data(self, filename):
        cur_filename = os.path.basename(filename).split(".")[0]
        # import pdb

        # pdb.set_trace()
        cachefile = Path(self.dataset_dir) / "cached" / (cur_filename + ".npz")
        data_out = dict()

        if cachefile.is_file():
            print("Loading cached data:", self.dataset_type)
            data = np.load(cachefile)
        else:
            print("There is no cached data:", self.dataset_type, "/ Generating data...")
            print(filename)
            with nc.Dataset(filename, "r") as f:
                for variable in f.variables:
                    if variable == "latitude":
                        lat = f.variables[variable][:]
                    elif variable == "longitude":
                        lon = f.variables[variable][:]
                    elif variable == "time":
                        time = f.variables[variable][:]
                    elif variable == "z":
                        # Fixed
                        target = f.variables[variable][:]  # (time, lat, lon)

                        # target = (
                        #     torch.tensor(f.variables[variable][0].data)
                        #     .unsqueeze(2)
                        #     .numpy()
                        # )

            time = (time - time.min()) / (time.max() - time.min())  # time normalization

            lats, lons = np.meshgrid(lat, lon)
            lats, lons = lats.T.data, lons.T.data  # (lat, lon), (lat, lon)
            xyz = sphere_to_cartesian(lats, lons)  # (lat, lon, 3)

            print("Triangulation")
            hull = ConvexHull(xyz.reshape(-1, 3))
            mesh = pymesh.form_mesh(hull.points, hull.simplices)
            points, adj = mesh_to_graph(mesh)

            print(f"Computing embeddings, size=({adj.shape})")
            u = get_fourier(adj)
            # if self.normalize:
            #     target = (target - target.min()) / (target.max() - target.min())
            os.makedirs(Path(self.dataset_dir) / "cached", exist_ok=True)
            np.savez(
                cachefile,
                points=points,
                fourier=u,
                target=target,
                time=time,
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

        data_out["time"] = torch.from_numpy(data["time"]).float()  # [t, 1]
        data_out["target"] = torch.from_numpy(data["target"]).float()  # [t,n, 1]

        time_resolution_index = np.arange(0, len(data["time"]), self.time_resolution)
        data_out["time"] = data_out["time"][time_resolution_index]
        data_out["target"] = data_out["target"][time_resolution_index]

        # Time sampling
        self.n_points = data["target"].shape[0]
        self.points_idx = np.arange(start, len(data_out["time"]), step)

        data_out["fourier"] = torch.from_numpy(data["fourier"]).float()  # [n, 100]
        data_out["fourier"] = data_out["fourier"][:, : self.n_fourier]  # [n/3, 34]
        data_out["time"] = data_out["time"][self.points_idx].view(-1, 1)  # [t, 1]
        data_out["target"] = data_out["target"][self.points_idx]  # [t/3,n, 1]
        data_out["target"] = data_out["target"].view(-1, 1)  # [t/3 * n, 1]
        # import pdb

        # pdb.set_trace()
        if self.zscore_normalize or self.normalize:
            self.scaler.fit(data_out["target"])
            data_out["target"] = self.scaler.transform(data_out["target"])

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
