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

# ─── allow importing your globals.py ─────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, DATA_DIR

# ─── paths ───────────────────────────────────────────────────────────────────
RESAMPLED_LAI     = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2001-2014.nc"
POINTS_CSV        = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
OUTPUT_NPZ_fast   = PROCESSED_DIR / "V_fast_2001-2014.npz"
OUTPUT_NPZ_slow   = PROCESSED_DIR / "V_slow_2001-2014.npz"
SHP_PATH          = DATA_DIR      / "Loess_Plateau_vector_border.shp"

# ─── load grid definition ────────────────────────────────────────────────────
df_pts     = pd.read_csv(POINTS_CSV)
lat_unique = np.sort(df_pts["LAT"].unique())[::-1]    # 844 rows, descending
lon_unique = np.sort(df_pts["LON"].unique())         # 1263 cols, ascending

# ─── load and prepare the Loess Plateau boundary ─────────────────────────────
border = gpd.read_file(SHP_PATH)
if border.crs is None:
    border.set_crs(epsg=4326, inplace=True)
elif border.crs.to_string() != "EPSG:4326":
    border = border.to_crs("EPSG:4326")

# build a 2D boolean mask of shape (n_lat, n_lon)
lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)
flat_pts = [Point(x, y) for x, y in zip(lon_grid.ravel(), lat_grid.ravel())]
pts_gs   = gpd.GeoSeries(flat_pts, crs="EPSG:4326")

# use shapely.ops.unary_union instead of the deprecated .unary_union on GeoDataFrame
poly     = unary_union(border.geometry)

mask_flat = pts_gs.within(poly).values
mask      = mask_flat.reshape(lat_unique.size, lon_unique.size)

# ─── open the resampled LAI file ─────────────────────────────────────────────
ds        = xr.open_dataset(RESAMPLED_LAI)
lai_vals  = ds["lai"].values.astype("float32")       # shape = (n_months, n_points)
times     = ds["time"].values                        # datetime64 array
ds.close()

# ─── vegetation‐input transforms ─────────────────────────────────────────────
def vegetation_input_fast(LAI_array: np.ndarray) -> np.ndarray:
    LAI_safe = np.maximum(LAI_array, 1e-6)
    return (0.1587 * np.log(LAI_safe) + 0.1331) * 0.8 * 8

def vegetation_input_slow(LAI_array: np.ndarray) -> np.ndarray:
    LAI_safe = np.maximum(LAI_array, 1e-6)
    return (0.1587 * np.log(LAI_safe) + 0.1331) * 0.2 * 8

# ─── pivot each month back to (844×1263), mask, and apply transforms ───────
n_months = lai_vals.shape[0]
V_fast   = np.empty((n_months, lat_unique.size, lon_unique.size), dtype="float32")
V_slow   = np.empty((n_months, lat_unique.size, lon_unique.size), dtype="float32")

for m in range(n_months):
    dfm = pd.DataFrame({
        "LAT": df_pts["LAT"],
        "LON": df_pts["LON"],
        "LAI": lai_vals[m, :]
    })
    grid = (
        dfm
        .pivot(index="LAT", columns="LON", values="LAI")
        .reindex(index=lat_unique, columns=lon_unique)
        .values
    )
    # mask everything outside the Loess Plateau
    grid[~mask] = np.nan

    V_fast[m] = vegetation_input_fast(grid)
    V_slow[m] = vegetation_input_slow(grid)

# ─── save V as its own compressed .npz ───────────────────────────────────────
np.savez_compressed(OUTPUT_NPZ_fast, v_fast=V_fast)
np.savez_compressed(OUTPUT_NPZ_slow, v_slow=V_slow)

print(f"✅ Saved V_fast matrix {V_fast.shape} to {OUTPUT_NPZ_fast}")
print(f"✅ Saved V_slow matrix {V_slow.shape} to {OUTPUT_NPZ_slow}")

