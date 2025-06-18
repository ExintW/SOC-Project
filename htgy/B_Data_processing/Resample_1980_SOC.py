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
# RESAMPLE SOC TO YOUR 1 KM GRID, AVERAGE OVER DEPTHS, SAVE AS 2D MATRIX & VISUALIZE
# -------------------------------------------------------------------------

# (a) Path to your SOM NetCDF
nc_path = DATA_DIR / "1980_SOM" / "SOM.nc"

# (b) Your 1 km CSV grid
csv_path = PROCESSED_DIR / "Resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"

# -------------------------------------------------------------------------
# 1) LOAD DATA
# -------------------------------------------------------------------------
ds = xr.open_dataset(nc_path)

depths = [4.5, 9.1, 16.6, 28.9]
conv_factor = 0.58 * 10  # SOM → SOC in g/kg

grid = pd.read_csv(csv_path, usecols=["LON", "LAT"])
lons = grid["LON"].values
lats = grid["LAT"].values

# -------------------------------------------------------------------------
# 2) INTERPOLATE EACH DEPTH
# -------------------------------------------------------------------------
for d in depths:
    da = ds.SOM.sel(depth=d, method="nearest") * conv_factor
    da = da.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    interp = da.interp(
        lon=xr.DataArray(lons, dims="points"),
        lat=xr.DataArray(lats, dims="points"),
        method="linear"
    ).values
    grid[f"soc_{d:.1f}cm"] = interp

# -------------------------------------------------------------------------
# 3) MEAN OVER DEPTHS (1D)
# -------------------------------------------------------------------------
soc_cols = [f"soc_{d:.1f}cm" for d in depths]
grid["soc_mean"] = grid[soc_cols].mean(axis=1)

# -------------------------------------------------------------------------
# 3b) PRINT AVERAGE OVER ALL POINTS
# -------------------------------------------------------------------------
avg_soc = grid["soc_mean"].mean()
print(f"Average SOC across all grid cells: {avg_soc:.4f} g/kg")

# -------------------------------------------------------------------------
# 4) RESHAPE INTO 2D MATRIX
# -------------------------------------------------------------------------
# Pivot so rows=unique LAT, cols=unique LON, values=soc_mean
pivot = grid.pivot(index="LAT", columns="LON", values="soc_mean")

lat_vals = pivot.index.values    # sorted unique latitudes
lon_vals = pivot.columns.values  # sorted unique longitudes
mean_mat = pivot.values          # 2D array shape (n_lat, n_lon)

# -------------------------------------------------------------------------
# 5) SAVE TO NPZ (2D)
# -------------------------------------------------------------------------
out_npz = PROCESSED_DIR / "soc_resampled_1980_matrix.npz"
np.savez(
    out_npz,
    lon=lon_vals,
    lat=lat_vals,
    soc_mean_matrix=mean_mat
)
print(f"Saved 2D SOC mean matrix to {out_npz} (shape {mean_mat.shape})")

# -------------------------------------------------------------------------
# 6) VISUALIZE MEAN‐SOC FIELD (2D)
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
# extent = [min_lon, max_lon, min_lat, max_lat]
extent = [lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()]
plt.imshow(
    mean_mat,
    origin="lower",
    extent=extent,
    vmin=0, vmax=60,
    cmap="viridis",
    aspect="auto"
)
plt.colorbar(label="Mean SOC (g/kg)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Mean SOC (4 depths) on 1 km Loess Plateau Grid")
plt.tight_layout()
plt.show()
