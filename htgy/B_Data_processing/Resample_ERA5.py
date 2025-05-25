import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    # ← added for plotting
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# Define paths
ERA5_DATA_DIR = DATA_DIR / "ERA5"
csv_file       = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
output_dir     = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled"

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

        # detect which time dimension (if any) we have
        if   "time"       in ds_res.dims: time_dim = "time"
        elif "valid_time" in ds_res.dims: time_dim = "valid_time"
        else:                             time_dim = None

        # --- your existing stats printout ---
        for var in ("lai_lv", "lai_hv", "tp"):
            if var not in ds_res:
                continue

            da = ds_res[var]

            if time_dim:
                # loop over each timestamp
                for t in da[time_dim].values:
                    arr = da.sel({time_dim: t})
                    dt    = pd.to_datetime(t)
                    month = dt.month
                    vmin, vmax, vmean = float(arr.min()), float(arr.max()), float(arr.mean())
                    print(f"{year}-{month:02d} | {var}: min={vmin:.3f}, max={vmax:.3f}, mean={vmean:.3f}")
            else:
                # no time axis: single snapshot
                vmin, vmax, vmean = float(da.min()), float(da.max()), float(da.mean())
                print(f"{year} | {var}: min={vmin:.3f}, max={vmax:.3f}, mean={vmean:.3f}")

        # save out the resampled NetCDF
        out_path = output_dir / f"resampled_{year}.nc"
        ds_res.to_netcdf(out_path)
        print(f"✅ Resampling complete for {year}. Saved to: {out_path}")

        # ──────────────── NEW: Annual‐mean LAI visualizations ────────────────
        if time_dim:
            for lai_var in ("lai_lv", "lai_hv"):
                if lai_var in ds_res:
                    annual_mean = ds_res[lai_var].mean(dim=time_dim)

                    plt.figure(figsize=(10, 6))
                    sc = plt.scatter(
                        lons, lats,
                        c=annual_mean,
                        s=10,
                        edgecolor="none"
                    )
                    plt.colorbar(sc, label=f"{lai_var} annual mean ({year})")
                    plt.xlabel("Longitude")
                    plt.ylabel("Latitude")
                    plt.title(f"Annual Mean {lai_var} in {year}")
                    plt.tight_layout()

                    fig_path = output_dir / f"annual_mean_{lai_var}_{year}.png"
                    plt.savefig(fig_path)
                    plt.close()
                    print(f"📊 Saved annual‐mean map: {fig_path.name}")

            # ───────────── NEW: Annual‐total Precipitation visual ─────────────
            if "tp" in ds_res:
                # sum over months to get annual total
                annual_tp = ds_res["tp"].sum(dim=time_dim)

                plt.figure(figsize=(10, 6))
                sc = plt.scatter(
                    lons, lats,
                    c=annual_tp,
                    s=10,
                    edgecolor="none"
                )
                plt.colorbar(sc, label=f"Annual total precipitation (tp) in {year}")
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plt.title(f"Annual Total Precipitation in {year}")
                plt.tight_layout()

                fig_path = output_dir / f"annual_total_tp_{year}.png"
                plt.savefig(fig_path)
                plt.close()
                print(f"📊 Saved annual‐total precipitation map: {fig_path.name}")
        # ────────────────────────────────────────────────────────────────────────

        ds.close()

    except Exception as e:
        print(f"❌ Error processing {year}: {e}")
