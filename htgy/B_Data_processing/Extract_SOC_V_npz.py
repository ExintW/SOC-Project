# =============================================================================
# 0) Imports & Globals
# =============================================================================
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# allow importing your globals.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import OUTPUT_DIR

# =============================================================================
# 1) Configuration & File-path Setup
# =============================================================================
present_dir     = OUTPUT_DIR / "Data" / "SOC_Present 5"
future_dir      = OUTPUT_DIR / "Data" / "SOC_Future 5" / "245"     # only ssp245
present_years   = list(range(2007, 2025))  # inclusive
future_years    = list(range(2025, 2101))  # inclusive

# =============================================================================
# 2) Helper to build a 3D array for one variable
# =============================================================================
def build_time_series(var_col, years, base_dir):
    """
    Reads month-by-month parquet files for all years in `years`,
    pivots each into a 2D grid (lat × lon), and returns a 3D ndarray
    of shape (n_months, n_lat, n_lon).
    """
    # figure out grid from first file
    sample_path = base_dir / f"SOC_terms_{years[0]}_01_River.parquet"
    df0 = pd.read_parquet(sample_path)
    lats = np.sort(df0["LAT"].unique())
    lons = np.sort(df0["LON"].unique())
    nlat, nlon = lats.size, lons.size
    ntime = len(years) * 12

    arr = np.empty((ntime, nlat, nlon), dtype=df0[var_col].dtype)
    idx = 0

    for yr in years:
        for mo in range(1, 13):
            path = base_dir / f"SOC_terms_{yr}_{mo:02d}_River.parquet"
            df   = pd.read_parquet(path)
            # pivot back into grid
            grid = (
                df
                .pivot(index="LAT", columns="LON", values=var_col)
                .loc[lats, lons]
                .values
            )
            arr[idx] = grid
            idx += 1

    return arr

# =============================================================================
# 3) Build & Save Present (2007–2024)
# =============================================================================
fast_present   = build_time_series("C_fast",      present_years, present_dir)
slow_present   = build_time_series("C_slow",      present_years, present_dir)
vfast_present  = build_time_series("Vegetation_fast", present_years, present_dir)
vslow_present  = build_time_series("Vegetation_slow", present_years, present_dir)

np.savez_compressed(
    OUTPUT_DIR / "Fast SOC year 2007-2024.npz", data=fast_present
)
np.savez_compressed(
    OUTPUT_DIR / "Slow SOC year 2007-2024.npz", data=slow_present
)
np.savez_compressed(
    OUTPUT_DIR / "V_fast_2007-2024.npz", data=vfast_present
)
np.savez_compressed(
    OUTPUT_DIR / "V_slow_2007-2024.npz", data=vslow_present
)

print("Saved present SOC stacks (2007–2024) to four NPZ files.")

# =============================================================================
# 4) Build & Save Future (2025–2100, ssp245 only)
# =============================================================================
fast_future  = build_time_series("C_fast",      future_years, future_dir)
slow_future  = build_time_series("C_slow",      future_years, future_dir)
vfast_future = build_time_series("Vegetation_fast", future_years, future_dir)
vslow_future = build_time_series("Vegetation_slow", future_years, future_dir)

np.savez_compressed(
    OUTPUT_DIR / "Fast SOC year 2025-2100.npz", data=fast_future
)
np.savez_compressed(
    OUTPUT_DIR / "Slow SOC year 2025-2100.npz", data=slow_future
)
np.savez_compressed(
    OUTPUT_DIR / "V_fast_2025-2100.npz", data=vfast_future
)
np.savez_compressed(
    OUTPUT_DIR / "V_slow_2025-2100.npz", data=vslow_future
)

print("Saved future SOC stacks (2025–2100, ssp245) to four NPZ files.")
