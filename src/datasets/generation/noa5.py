import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import shutil
from datetime import datetime, timedelta

import getgfs
import imageio
import numpy as np
import plotly.graph_objects as go
import pymesh
from pytorch_lightning import seed_everything
from scipy.spatial import ConvexHull


seed_everything(1234)

# Fetch all data
# 'dpt2m'   : 2 m above ground dew point temperature [k]
# 'tcdcclm' : entire atmosphere total cloud cover [%]
# 'gustsfc' : surface wind speed (gust) [m/s]
print("Fetching data")
variables = ["dpt2m", "tcdcclm", "gustsfc"]
colorscale = ["hot", "Blues", "Spectral"]
# date = (datetime.now() - timedelta(1)).strftime("%Y%m%d")  # We used: "20220509"
date = "20230925"
lat_range = "[-90:90]"
lon_range = "[0:360]"
# resolution = "1p00"
resolution = "0p25"
f = getgfs.Forecast(resolution)
result = f.get(variables, f"{date} 00:00", lat_range, lon_range)

import pdb

pdb.set_trace()

# Get mesh
lat, lon = (
    np.array(result.variables[variables[0]].coords["lat"].values),
    np.array(result.variables[variables[0]].coords["lon"].values),
)
print(lat)
print(lon)
