import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rioxarray         # ← install via: pip install rioxarray
import xarray as xr
from pathlib import Path

# allow imports from parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# ─────────── File Paths ───────────
tiff_k1_path    = DATA_DIR  / "k1_halfDegree.tif"
tiff_k2_path    = DATA_DIR  / "k2_halfDegree.tif"
csv_file_path   = PROCESSED_DIR / "Resampled_Loess_Plateau_1km_with_DEM_region_labeled.csv"
output_csv_path = PROCESSED_DIR / "Resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
os.makedirs(output_csv_path.parent, exist_ok=True)

# ─────────── Helper: two‐step interpolation via rioxarray with nodata masking ───────────
def interp_tiff(path, lons, lats):
    da = rioxarray.open_rasterio(path)
    nodata = da.rio.nodata
    da = da.where(da != nodata)               # mask out sentinel nodata values
    da = da.squeeze("band", drop=True)
    da = da.rename({"x": "lon", "y": "lat"})
    # first linear…
    da_lin = da.interp(
        lon = xr.DataArray(lons, dims="points"),
        lat = xr.DataArray(lats, dims="points"),
        method = "linear"
    )
    # then nearest to fill any NaNs
    da_nn = da.interp(
        lon = xr.DataArray(lons, dims="points"),
        lat = xr.DataArray(lats, dims="points"),
        method = "nearest"
    )
    da_filled = da_lin.fillna(da_nn)
    return da_filled.values

# ─────────── SOM→SOC conversion (clip negatives to zero) ───────────
def convert_som_to_soc_monthly(som_k_day):
    k_day = np.maximum(som_k_day, 0.0)
    som_k_month = 1 - np.exp(-k_day * 30)
    return som_k_month * 0.58

# ─────────── Read input CSV ───────────
df = pd.read_csv(csv_file_path)
lon_csv = df["LON"].values
lat_csv = df["LAT"].values

# ─────────── Interpolate k₁ & k₂ ───────────
som_k1_day = interp_tiff(tiff_k1_path, lon_csv, lat_csv)
som_k2_day = interp_tiff(tiff_k2_path, lon_csv, lat_csv)

# ─────────── Fill negative SOM rates with mean of positives ───────────
mean_k1 = np.nanmean(som_k1_day[som_k1_day > 0])
mean_k2 = np.nanmean(som_k2_day[som_k2_day > 0])
som_k1_day = np.where(som_k1_day < 0, mean_k1, som_k1_day)
som_k2_day = np.where(som_k2_day < 0, mean_k2, som_k2_day)

# ─────────── Add new columns ───────────
df["SOC_k1_fast_pool (1/day)"]   = som_k1_day * 0.58
df["SOC_k2_slow_pool (1/day)"]   = som_k2_day * 0.58
df["SOC_k1_fast_pool (1/month)"] = convert_som_to_soc_monthly(som_k1_day)
df["SOC_k2_slow_pool (1/month)"] = convert_som_to_soc_monthly(som_k2_day)

# ─────────── Save updated CSV ───────────
df.to_csv(output_csv_path, index=False)
print(f"✅ Updated CSV saved: {output_csv_path}")

# ─────────── Plot spatial maps for k₁ & k₂ ───────────
plots = [
    ("SOC_k1_fast_pool (1/month)", "k₁ fast‐pool rate (1/month)", "map_k1_fast_pool.png"),
    ("SOC_k2_slow_pool (1/month)", "k₂ slow‐pool rate (1/month)", "map_k2_slow_pool.png"),
]

for col, label, fname in plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        lon_csv, lat_csv,
        c = df[col],
        s = 10,
        edgecolor = "none"
    )
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial distribution of {label}")
    # crop to data extent
    ax.set_xlim(lon_csv.min(), lon_csv.max())
    ax.set_ylim(lat_csv.min(), lat_csv.max())
    ax.margins(0)
    plt.tight_layout()
    fig_path = output_csv_path.parent / fname
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"📊 Saved map: {fig_path.name}")
