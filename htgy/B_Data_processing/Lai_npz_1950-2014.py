#!/usr/bin/env python3
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ─── allow importing your globals.py ─────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR

# ─── paths ───────────────────────────────────────────────────────────────────
RESAMPLED_LAI = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2001-2014.nc"
POINTS_CSV    = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
OUTPUT_NPZ_fast    = PROCESSED_DIR / "V_fast_2001-2014.npz"
OUTPUT_NPZ_slow    = PROCESSED_DIR / "V_slow_2001-2014.npz"

# ─── load grid definition ────────────────────────────────────────────────────
df_pts     = pd.read_csv(POINTS_CSV)
lat_unique = np.sort(df_pts["LAT"].unique())[::-1]    # 844 rows
lon_unique = np.sort(df_pts["LON"].unique())         # 1263 cols

# ─── open the resampled LAI file ─────────────────────────────────────────────
ds        = xr.open_dataset(RESAMPLED_LAI)
lai_vals  = ds["lai"].values.astype("float32")       # shape = (n_months, n_points)
times     = ds["time"].values                        # datetime64 array
ds.close()

# ─── vegetation‐input transform ──────────────────────────────────────────────
def vegetation_input_fast(LAI_array: np.ndarray) -> np.ndarray:
    LAI_safe = np.maximum(LAI_array, 1e-6)
    return (0.11434652 * np.log(LAI_safe) + 0.08709953)*0.8

def vegetation_input_slow(LAI_array: np.ndarray) -> np.ndarray:
    LAI_safe = np.maximum(LAI_array, 1e-6)
    return (0.11434652 * np.log(LAI_safe) + 0.08709953)*0.2

# ─── pivot each month back to (844×1263) and apply transform ───────────────
n_months = lai_vals.shape[0]
V_fast        = np.empty((n_months, lat_unique.size, lon_unique.size), dtype="float32")
V_slow        = np.empty((n_months, lat_unique.size, lon_unique.size), dtype="float32")

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
    V_fast[m] = vegetation_input_fast(grid)
    V_slow[m] = vegetation_input_slow(grid)

# ─── save V as its own compressed .npz ───────────────────────────────────────
np.savez_compressed(
    OUTPUT_NPZ_fast,
    v_fast=V_fast
)
np.savez_compressed(
    OUTPUT_NPZ_slow,
    v_slow=V_slow
)
print(f"✅ Saved V matrix {V_fast.shape} to {OUTPUT_NPZ_fast}")
print(f"✅ Saved V matrix {V_slow.shape} to {OUTPUT_NPZ_slow}")
