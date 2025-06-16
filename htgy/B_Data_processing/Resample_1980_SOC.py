import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rioxarray                   # registers the .rio accessor
from pathlib import Path

# Append project root so we can import DATA_DIR, PROCESSED_DIR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# -------------------------------------------------------------------------
# RESAMPLE SOC TO YOUR 1 KM GRID, AVERAGE OVER DEPTHS, SAVE & VISUALIZE
# -------------------------------------------------------------------------

# (a) Path to your SOM NetCDF
nc_path = DATA_DIR / "1980_SOM" / "SOM.nc"

# (b) Your 1 km CSV grid
csv_path = PROCESSED_DIR / "Resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"

# -------------------------------------------------------------------------
# 1) LOAD DATA
# -------------------------------------------------------------------------
ds = xr.open_dataset(nc_path)

# depths (cm) and SOC conversion factor
depths = [4.5, 9.1, 16.6, 28.9]
conv_factor = 0.58 * 10  # SOM → SOC in g/kg

# load grid
grid = pd.read_csv(csv_path, usecols=["LON", "LAT"])
lons = grid["LON"].values
lats = grid["LAT"].values

# -------------------------------------------------------------------------
# 2) INTERPOLATE EACH DEPTH
# -------------------------------------------------------------------------
for d in depths:
    # select SOM layer at depth d, convert to SOC
    da = ds.SOM.sel(depth=d, method="nearest") * conv_factor

    # assign CRS & spatial dims so rioxarray can interpolate
    da = (
        da
        .rio.write_crs("EPSG:4326")
        .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    )

    # interpolate linearly at each point
    interp_vals = da.interp(
        lon=xr.DataArray(lons, dims="points"),
        lat=xr.DataArray(lats, dims="points"),
        method="linear"
    ).values

    # add interpolated values to DataFrame
    grid[f"soc_{d:.1f}cm"] = interp_vals

# -------------------------------------------------------------------------
# 3) MEAN OVER DEPTHS
# -------------------------------------------------------------------------
soc_cols = [f"soc_{d:.1f}cm" for d in depths]
grid["soc_mean"] = grid[soc_cols].mean(axis=1)

# -------------------------------------------------------------------------
# 3b) PRINT GRID AVERAGE SOC
# -------------------------------------------------------------------------
avg_soc = grid["soc_mean"].mean()
print(f"Average SOC across all grid cells: {avg_soc:.4f} g/kg")

# -------------------------------------------------------------------------
# 4) SAVE TO NPZ
# -------------------------------------------------------------------------
out_npz = PROCESSED_DIR / "soc_resampled_1980.npz"
np.savez(
    out_npz,
    lon=grid["LON"].values,
    lat=grid["LAT"].values,
    **{col: grid[col].values for col in soc_cols + ["soc_mean"]}
)
print(f"Saved interpolated SOC fields to {out_npz}")

# -------------------------------------------------------------------------
# 5) VISUALIZE MEAN‐SOC FIELD
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sc = plt.scatter(
    grid["LON"], grid["LAT"],
    c=grid["soc_mean"],
    vmin=0, vmax=60,
    s=10,
    cmap="viridis"
)
plt.colorbar(sc, label="Mean SOC (g/kg)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Mean SOC (4 depths) on 1 km Loess Plateau Grid")
plt.tight_layout()
plt.show()
