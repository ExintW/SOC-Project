import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    # ← for plotting

# Project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# Define new paths for soil temperature
ERA5_DATA_DIR = DATA_DIR / "ERA5_Temp"  # ← CHANGED to ERA5_temp
csv_file      = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
output_dir    = PROCESSED_DIR / "ERA5_data_Temp_Monthly_Resampled"

os.makedirs(output_dir, exist_ok=True)

# Load point coordinates
df_points = pd.read_csv(csv_file)
lons = df_points["LON"].values
lats = df_points["LAT"].values

start_year = 1950
end_year   = 2025

for year in range(start_year, end_year + 1):
    nc_file = ERA5_DATA_DIR / f"{year}.nc"
    if not nc_file.exists():
        print(f"File {nc_file} not found, skipping...")
        continue

    try:
        ds = xr.open_dataset(nc_file)

        # detect lon/lat names
        lon_name = "longitude" if "longitude" in ds.dims else "lon"
        lat_name = "latitude"  if "latitude"  in ds.dims else "lat"

        # linear interpolation onto your points
        ds_res = ds.interp(
            {
                lon_name: xr.DataArray(lons, dims="points"),
                lat_name: xr.DataArray(lats, dims="points")
            },
            method="linear"
        )

        # detect time dimension
        if   "time"       in ds_res.dims: time_dim = "time"
        elif "valid_time" in ds_res.dims: time_dim = "valid_time"
        else:                             time_dim = None

        # --- stats printout for stl1 and stl2 ---
        for var in ("stl1", "stl2"):
            if var not in ds_res:
                continue

            da = ds_res[var]

            if time_dim:
                for t in da[time_dim].values:
                    arr = da.sel({time_dim: t})
                    dt    = pd.to_datetime(t)
                    month = dt.month
                    vmin, vmax, vmean = float(arr.min()), float(arr.max()), float(arr.mean())
                    print(f"{year}-{month:02d} | {var}: min={vmin:.3f}, max={vmax:.3f}, mean={vmean:.3f}")
            else:
                vmin, vmax, vmean = float(da.min()), float(da.max()), float(da.mean())
                print(f"{year} | {var}: min={vmin:.3f}, max={vmax:.3f}, mean={vmean:.3f}")

        # save NetCDF
        out_path = output_dir / f"resampled_{year}.nc"
        ds_res.to_netcdf(out_path)
        print(f"✅ Resampling complete for {year}. Saved to: {out_path}")

        # ─────── Annual Mean Temperature Visualizations ───────
        if time_dim:
            for temp_var in ("stl1", "stl2"):
                if temp_var in ds_res:
                    annual_mean = ds_res[temp_var].mean(dim=time_dim)

                    plt.figure(figsize=(10, 6))
                    sc = plt.scatter(
                        lons, lats,
                        c=annual_mean,
                        s=10,
                        edgecolor="none"
                    )
                    plt.colorbar(sc, label=f"{temp_var} annual mean ({year})")
                    plt.xlabel("Longitude")
                    plt.ylabel("Latitude")
                    plt.title(f"Annual Mean {temp_var.upper()} in {year}")
                    plt.tight_layout()

                    fig_path = output_dir / f"annual_mean_{temp_var}_{year}.png"
                    plt.savefig(fig_path)
                    plt.close()
                    print(f"📊 Saved annual‐mean map: {fig_path.name}")
        # ────────────────────────────────────────────────

        ds.close()

    except Exception as e:
        print(f"❌ Error processing {year}: {e}")
