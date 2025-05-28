#!/usr/bin/env python3
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ─── 0) make sure you can import your globals.py ─────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR

# ─── 1) paths & parameters ─────────────────────────────────────────────────
CSV_FILE      = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
RESAMPLED_DIR = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled"
# this will be a raw memmap on disk:
MEMMAP_FILE   = PROCESSED_DIR / "tp_1950-2014.dat"
# optional compressed bundle:
NPZ_FILE      = PROCESSED_DIR / "tp_1950-2014.npz"

# get your list of resampled_*.nc (should be 100 years → ~100 files)
nc_files = sorted(RESAMPLED_DIR.glob("resampled_*.nc"))
n_years  = len(nc_files)
if n_years == 0:
    raise FileNotFoundError(f"No resampled_*.nc in {RESAMPLED_DIR!r}")

# load your point‐coordinates CSV
df_pts = pd.read_csv(CSV_FILE)
# get the unique grid axes
lat_unique = np.sort(df_pts["LAT"].unique())[::-1]  # descending so row 0 is top
lon_unique = np.sort(df_pts["LON"].unique())       # ascending so col 0 is left

nlat, nlon = lat_unique.size, lon_unique.size
n_months_per_file = xr.open_dataset(nc_files[0])["tp"].sizes["valid_time"]
total_months = n_years * n_months_per_file  # e.g. 100 × 12 = 1200

# ─── 2) create a float32 memmap of the right shape ─────────────────────────
tp_mm = np.memmap(
    str(MEMMAP_FILE),
    dtype="float32",
    mode="w+",
    shape=(total_months, nlat, nlon)
)

# ─── 3) loop & fill ────────────────────────────────────────────────────────
idx = 0
for nc in nc_files:
    ds = xr.open_dataset(nc)
    # this is (time, points)
    tp_vals = ds["tp"].values.astype("float32")
    ds.close()

    # for each month, pivot back to a grid and write into the memmap
    for m in range(tp_vals.shape[0]):
        tmp = pd.DataFrame({
            "LAT": df_pts["LAT"],
            "LON": df_pts["LON"],
            "tp":  tp_vals[m, :]
        })
        grid = tmp.pivot(index="LAT", columns="LON", values="tp")
        # reindex ensures it's exactly (nlat × nlon) in the right order
        grid = grid.reindex(index=lat_unique, columns=lon_unique)
        tp_mm[idx, :, :] = grid.values
        idx += 1

# flush to disk so nothing sits in RAM
tp_mm.flush()
print(f"✅  Wrote memmap to {MEMMAP_FILE} (shape = {tp_mm.shape})")

# ─── 4) (Optional) bundle into a .npz ───────────────────────────────────────
# Note: this will still *read* from disk in chunks, not balloon into RAM.
np.savez_compressed(
    NPZ_FILE,
    tp  = tp_mm
)
print(f"✅  Also wrote compressed archive to {NPZ_FILE}")

