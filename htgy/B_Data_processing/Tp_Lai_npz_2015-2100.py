#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

# allow importing your globals.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, DATA_DIR

# ─── paths ───────────────────────────────────────────────────────────────────
CSV_FILE    = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
CMIP6_DIR   = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled"
LAI_FILE    = CMIP6_DIR / "resampled_lai_points_2015-2100_245.nc"
PR_FILE     = CMIP6_DIR / "resampled_pr_points_2015-2100_245.nc"
SHP_PATH    = DATA_DIR      / "Loess_Plateau_vector_border.shp"

V_FAST_NPZ  = PROCESSED_DIR / "V_fast_245_2015-2100.npz"
V_SLOW_NPZ  = PROCESSED_DIR / "V_slow_245_2015-2100.npz"
PR_NPZ      = PROCESSED_DIR / "PR_245_2015-2100.npz"

# ─── build grid axes from your points CSV ─────────────────────────────────────
df_pts     = pd.read_csv(CSV_FILE)
lat_unique = np.sort(df_pts["LAT"].unique())[::-1]  # descending → row 0 = northmost
lon_unique = np.sort(df_pts["LON"].unique())       # ascending  → col 0 = westmost
nlat, nlon = lat_unique.size, lon_unique.size

# ─── load & prepare the Loess Plateau boundary mask ─────────────────────────
border = gpd.read_file(SHP_PATH)
if border.crs is None:
    border.set_crs(epsg=4326, inplace=True)
elif border.crs.to_string() != "EPSG:4326":
    border = border.to_crs("EPSG:4326")

# make a flat list of Points and test within the unified polygon
lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)
flat_pts = [Point(x, y) for x, y in zip(lon_grid.ravel(), lat_grid.ravel())]
poly     = unary_union(border.geometry)
mask_flat = gpd.GeoSeries(flat_pts, crs="EPSG:4326").within(poly).values
mask      = mask_flat.reshape(nlat, nlon)  # True inside, False outside

# ─── vegetation‐input transforms ─────────────────────────────────────────────
def vegetation_input_fast(LAI_array):
    LAI_safe = np.maximum(LAI_array, 1e-6)
    return (0.1587 * np.log(LAI_safe) + 0.1331) * 0.8 * 8

def vegetation_input_slow(LAI_array):
    LAI_safe = np.maximum(LAI_array, 1e-6)
    return (0.1587 * np.log(LAI_safe) + 0.1331) * 0.2 * 8

# ─── Process LAI → V_fast & V_slow ────────────────────────────────────────────
with xr.open_dataset(LAI_FILE) as ds_lai:
    lai_vals = ds_lai["lai"].values.astype("float32")  # (time, points)

n_months = lai_vals.shape[0]
V_fast   = np.empty((n_months, nlat, nlon), dtype="float32")
V_slow   = np.empty((n_months, nlat, nlon), dtype="float32")

for m in range(n_months):
    tmp = pd.DataFrame({
        "LAT": df_pts["LAT"],
        "LON": df_pts["LON"],
        "VAL": lai_vals[m, :]
    })
    grid = (
        tmp
        .pivot(index="LAT", columns="LON", values="VAL")
        .reindex(index=lat_unique, columns=lon_unique)
        .values
    )
    # compute V and then zero out-of-bounds
    vf = vegetation_input_fast(grid)
    vs = vegetation_input_slow(grid)
    vf[~mask] = np.nan
    vs[~mask] = np.nan

    V_fast[m] = vf
    V_slow[m] = vs

# ─── save V arrays ───────────────────────────────────────────────────────────
np.savez_compressed(V_FAST_NPZ, v_fast=V_fast)
print(f"✅ Saved V_fast matrix with shape {V_fast.shape} to {V_FAST_NPZ}")

np.savez_compressed(V_SLOW_NPZ, v_slow=V_slow)
print(f"✅ Saved V_slow matrix with shape {V_slow.shape} to {V_SLOW_NPZ}")

# ─── Process PR → PR array ───────────────────────────────────────────────────
with xr.open_dataset(PR_FILE) as ds_pr:
    pr_var  = "pr" if "pr" in ds_pr.data_vars else list(ds_pr.data_vars)[0]
    pr_vals = ds_pr[pr_var].values.astype("float32")  # (time, points)

PR = np.empty((pr_vals.shape[0], nlat, nlon), dtype="float32")
for m in range(pr_vals.shape[0]):
    tmp = pd.DataFrame({
        "LAT": df_pts["LAT"],
        "LON": df_pts["LON"],
        "VAL": pr_vals[m, :]
    })
    grid = (
        tmp
        .pivot(index="LAT", columns="LON", values="VAL")
        .reindex(index=lat_unique, columns=lon_unique)
        .values
    )
    # zero out-of-bounds precipitation
    grid[~mask] = np.nan
    PR[m] = grid

# ─── save PR array ───────────────────────────────────────────────────────────
np.savez_compressed(PR_NPZ, Precip=PR)
print(f"✅ Saved PR matrix with shape {PR.shape} to {PR_NPZ}")
