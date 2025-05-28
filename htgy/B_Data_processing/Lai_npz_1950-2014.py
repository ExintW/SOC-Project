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
RESAMPLED_LAI = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_1950-2000.nc"
POINTS_CSV    = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
OUTPUT_NPZ    = PROCESSED_DIR / "V_1950-2000.npz"

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
def vegetation_input(LAI_array: np.ndarray) -> np.ndarray:
    LAI_safe = np.maximum(LAI_array, 1e-6)
    return 0.11434652 * np.log(LAI_safe) + 0.08709953

# ─── pivot each month back to (844×1263) and apply transform ───────────────
n_months = lai_vals.shape[0]
V        = np.empty((n_months, lat_unique.size, lon_unique.size), dtype="float32")

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
    V[m] = vegetation_input(grid)

# ─── save V as its own compressed .npz ───────────────────────────────────────
np.savez_compressed(
    OUTPUT_NPZ,
    V=V
)

print(f"✅ Saved V matrix {V.shape} to {OUTPUT_NPZ}")
