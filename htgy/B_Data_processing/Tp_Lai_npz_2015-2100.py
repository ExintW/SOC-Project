#!/usr/bin/env python3
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# allow importing your globals.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR

# Paths
CSV_FILE  = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
CMIP6_DIR = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled"
LAI_FILE  = CMIP6_DIR / "resampled_lai_points_2015-2100_245.nc"
PR_FILE   = CMIP6_DIR / "resampled_pr_points_2015-2100_245.nc"

V_fast_NPZ     = PROCESSED_DIR / "V_fast_245_2015-2100.npz"
V_slow_NPZ     = PROCESSED_DIR / "V_slow_245_2015-2100.npz"
PR_NPZ    = PROCESSED_DIR / "PR_245_2015-2100.npz"

# Build grid axes from your points CSV
df_pts    = pd.read_csv(CSV_FILE)
lat_unique = np.sort(df_pts["LAT"].unique())[::-1]  # descending → row 0 = northmost
lon_unique = np.sort(df_pts["LON"].unique())       # ascending  → col 0 = westmost
nlat, nlon = lat_unique.size, lon_unique.size

# Vegetation‐input transform
def vegetation_input_fast(LAI_array):
    LAI_safe = np.maximum(LAI_array, 1e-6)
    return (0.11434652 * np.log(LAI_safe) + 0.08709953)*0.2

def vegetation_input_slow(LAI_array):
    LAI_safe = np.maximum(LAI_array, 1e-6)
    return (0.11434652 * np.log(LAI_safe) + 0.08709953)*0.8

# ——— Process LAI → V —————————————————————————————————————————————————————
with xr.open_dataset(LAI_FILE) as ds_lai:
    lai_vals = ds_lai["lai"].values.astype("float32")  # (time, points)
n_months = lai_vals.shape[0]
V_fast = np.empty((n_months, nlat, nlon), dtype="float32")
V_slow = np.empty((n_months, nlat, nlon), dtype="float32")

for m in range(n_months):
    tmp = pd.DataFrame({
        "LAT": df_pts["LAT"],
        "LON": df_pts["LON"],
        "VAL": lai_vals[m, :]
    })
    grid = (
        tmp.pivot(index="LAT", columns="LON", values="VAL")
           .reindex(index=lat_unique, columns=lon_unique)
           .values
    )
    V_fast[m] = vegetation_input_fast(grid)
    V_slow[m] = vegetation_input_slow(grid)

# Save V
np.savez_compressed(
    V_fast_NPZ,
    v_fast=V_fast
)
print(f"✅ Saved V_fast matrix with shape {V_fast.shape} to {V_fast_NPZ}")
np.savez_compressed(
    V_slow_NPZ,
    v_slow=V_slow
)
print(f"✅ Saved V_fast matrix with shape {V_slow.shape} to {V_slow_NPZ}")

# ——— Process PR —————————————————————————————————————————————————————————
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
        tmp.pivot(index="LAT", columns="LON", values="VAL")
           .reindex(index=lat_unique, columns=lon_unique)
           .values
    )
    PR[m] = grid

# Save PR
np.savez_compressed(
    PR_NPZ,
    Precip=PR
)
print(f"✅ Saved PR matrix with shape {PR.shape} to {PR_NPZ}")

