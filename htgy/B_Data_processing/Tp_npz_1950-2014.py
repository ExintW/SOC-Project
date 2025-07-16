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

# ─── 0) make sure you can import your globals.py ─────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, DATA_DIR

# ─── 1) paths & parameters ─────────────────────────────────────────────────
CSV_FILE      = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
RESAMPLED_DIR = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled"
SHP_PATH      = DATA_DIR      / "Loess_Plateau_vector_border.shp"

# this will be a raw memmap on disk:
MEMMAP_FILE   = PROCESSED_DIR / "tp_1950-2024.dat"
# optional compressed bundle:
NPZ_FILE      = PROCESSED_DIR / "tp_1950-2024.npz"

# ─── 2) load your point‐coordinates CSV & build grid axes ─────────────────────
df_pts     = pd.read_csv(CSV_FILE)
lat_unique = np.sort(df_pts["LAT"].unique())[::-1]  # descending so row 0 is top
lon_unique = np.sort(df_pts["LON"].unique())       # ascending so col 0 is left
nlat, nlon = lat_unique.size, lon_unique.size

# ─── 3) build the Loess Plateau mask ────────────────────────────────────────
border = gpd.read_file(SHP_PATH)
if border.crs is None:
    border.set_crs(epsg=4326, inplace=True)
elif border.crs.to_string() != "EPSG:4326":
    border = border.to_crs("EPSG:4326")

lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)
flat_pts = [Point(x, y) for x, y in zip(lon_grid.ravel(), lat_grid.ravel())]
poly     = unary_union(border.geometry)
mask_flat = gpd.GeoSeries(flat_pts, crs="EPSG:4326").within(poly).values
mask      = mask_flat.reshape(nlat, nlon)  # True inside, False outside

# ─── 4) list NC files & prepare memmap ──────────────────────────────────────
nc_files = sorted(RESAMPLED_DIR.glob("resampled_*.nc"))
n_years  = len(nc_files)
if n_years == 0:
    raise FileNotFoundError(f"No resampled_*.nc in {RESAMPLED_DIR!r}")
# assume each file has the same number of months
n_months_per = xr.open_dataset(nc_files[0])["tp"].sizes["valid_time"]
total_months = n_years * n_months_per

tp_mm = np.memmap(
    str(MEMMAP_FILE),
    dtype="float32",
    mode="w+",
    shape=(total_months, nlat, nlon)
)

# ─── 5) loop over each file & month, pivot, mask, write ────────────────────
idx = 0
for nc in nc_files:
    ds = xr.open_dataset(nc)
    tp_vals = ds["tp"].values.astype("float32")  # (time, points)
    ds.close()

    for m in range(tp_vals.shape[0]):
        tmp = pd.DataFrame({
            "LAT": df_pts["LAT"],
            "LON": df_pts["LON"],
            "tp":  tp_vals[m, :]
        })
        grid = (
            tmp.pivot(index="LAT", columns="LON", values="tp")
               .reindex(index=lat_unique, columns=lon_unique)
               .values
        )
        # zero out anything outside the Loess Plateau
        grid[~mask] = np.nan

        tp_mm[idx, :, :] = grid
        idx += 1

# flush to disk so nothing sits in RAM
tp_mm.flush()
print(f"✅  Wrote memmap to {MEMMAP_FILE} (shape = {tp_mm.shape})")

# ─── 6) (Optional) bundle into a .npz ───────────────────────────────────────
# Note: this will still read from disk in chunks, not balloon into RAM.
np.savez_compressed(
    NPZ_FILE,
    precip=tp_mm
)
print(f"✅  Also wrote compressed archive to {NPZ_FILE}")
