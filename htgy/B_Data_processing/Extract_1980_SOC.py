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

# ---------- 2) EXTRACT EACH DEPTH (no interpolation, nearest with tolerance) ----------
# infer source-grid native spacing to define a safe tolerance (half cell)
lon_res = float(np.median(np.diff(ds.lon.values))) if ds.lon.size > 1 else 0.0
lat_res = float(np.median(np.diff(ds.lat.values))) if ds.lat.size > 1 else 0.0
tol = 0.5 * max(abs(lon_res), abs(lat_res)) if (lon_res != 0.0 and lat_res != 0.0) else None

for d in depths:
    da = ds.SOM.sel(depth=d, method="nearest") * conv_factor
    da = da.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    vals = []
    for lon, lat in zip(lons, lats):
        try:
            if tol is None:
                # exact match only; if label not found -> KeyError -> NaN
                val = da.sel(lon=lon, lat=lat).values.item()
            else:
                # nearest WITH tolerance; if nearest is farther than tol -> KeyError -> NaN
                val = da.sel(lon=lon, lat=lat, method="nearest",
                             tolerance=tol).values.item()
        except Exception:
            val = np.nan
        vals.append(val)
    grid[f"soc_{d:.1f}cm"] = vals
    
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
out_npz = PROCESSED_DIR / "soc_1980_matrix.npz"
np.savez(
    out_npz,
    lon=lon_vals,
    lat=lat_vals,
    soc_mean_matrix=mean_mat
)
print(f"Saved 2D SOC mean matrix to {out_npz} (shape {mean_mat.shape})")

# ---------- 6) VISUALIZE MEAN‐SOC FIELD (2D) ----------
plt.figure(figsize=(10, 6))
extent = [lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()]

# make NaNs transparent so missing pixels stay blank
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad(alpha=0)

plt.imshow(
    np.ma.masked_invalid(mean_mat),  # mask NaNs explicitly
    origin="lower",
    extent=extent,
    vmin=0, vmax=60,
    cmap=cmap,
    aspect="auto"
)
plt.colorbar(label="Mean SOC (g/kg)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Mean SOC (4 depths) on 1 km Loess Plateau Grid")
plt.tight_layout()
plt.show()
