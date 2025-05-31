import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import csv
from pathlib import Path

# Append project root so we can import DATA_DIR, PROCESSED_DIR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# Input file paths for CMIP6 LAI and PR
lai_path = Path(DATA_DIR) / "CMIP6" / "lai_Lmon_BCC-CSM2-MR_ssp370_r1i1p1f1_gn_201501-210012.nc"
pr_path  = Path(DATA_DIR) / "CMIP6" / "pr_Amon_BCC-CSM2-MR_ssp370_r1i1p1f1_gn_201501-210012.nc"

# Load target 1 km grid points
csv_pts = Path(PROCESSED_DIR) / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
df_pts  = pd.read_csv(csv_pts)
lons     = df_pts["LON"].values
lats     = df_pts["LAT"].values

# Output directory for interpolated data and stats
output_dir = Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled"
os.makedirs(output_dir, exist_ok=True)


def interp_and_collect(ds, var_name, label, save_interp_fname):
    """
    Interpolate ds[var_name] onto target points bilinearly,
    save the interpolated dataset, and return annual stats rows.
    """
    # Determine coordinate dimension names
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    lat_name = "latitude"  if "latitude"  in ds.dims else "lat"

    # 1) Bilinear interpolation onto the 1 km points
    ds_interp = ds.interp(
        {lon_name: xr.DataArray(lons, dims="points"),
         lat_name: xr.DataArray(lats, dims="points")},
        method="linear"
    )

    # 2) Save the interpolated Dataset to NetCDF
    interp_path = output_dir / save_interp_fname
    ds_interp.to_netcdf(interp_path)
    print(f"→ Saved interpolated {label} to {interp_path.name}")

    # 3) Compute annual min/max/mean across months and points
    da = ds_interp[var_name]
    rows = []
    for year, grp in da.groupby("time.year"):
        arr = grp.values.ravel()
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        mu = np.nanmean(arr)
        print(f"{label} {int(year)} → min={mn:.3f}, max={mx:.3f}, mean={mu:.3f}")
        rows.append((label, int(year), mn, mx, mu))

    ds_interp.close()
    return rows


# Collect statistics for both variables
lai_rows = []
pr_rows  = []

# Process LAI
try:
    ds_lai = xr.open_dataset(lai_path)
    print(">>> Processing LAI")
    lai_rows = interp_and_collect(
        ds_lai, "lai", "LAI", "resampled_lai_points_2015-2100_370.nc"
    )
    ds_lai.close()
except Exception as e:
    print(f"Error processing LAI: {e}")

# Process Precipitation
try:
    ds_pr = xr.open_dataset(pr_path)  # add engine='netcdf4' or 'h5netcdf' if needed
    pr_var = "pr" if "pr" in ds_pr.data_vars else list(ds_pr.data_vars)[0]
    print("\n>>> Processing PR")
    pr_rows = interp_and_collect(
        ds_pr, pr_var, "PR", "resampled_pr_points_2015-2100_370.nc"
    )
    ds_pr.close()
except Exception as e:
    print(f"Error processing PR: {e}")

# Write annual stats to CSV
stats_file = output_dir / "annual_CMIP6_stats.csv"
with open(stats_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["variable", "year", "min", "max", "mean"])
    for row in lai_rows + pr_rows:
        writer.writerow(row)
print(f"→ Annual stats saved to {stats_file.name}")

# ────────────────────────────────────────────────────────────────────────────
# 4) Visualization: spatial map of mean LAI and PR (2015–2100, SSP5-8.5)
# ────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt   # make sure this is at the top of your file

for var_label, fname, var_name, cb_label in [
    ("LAI", "resampled_lai_points_2015-2100_370.nc", "lai", "Mean LAI"),
    ("PR",  "resampled_pr_points_2015-2100_370.nc",  "pr",  "Mean Precipitation (kg/m²/s)")
]:
    # 1) open the interpolated file
    ds_vis = xr.open_dataset(output_dir / fname)

    # 2) compute the long‐term mean
    mean_da = ds_vis[var_name].mean(dim="time")

    # 3) grab lon/lat coords
    if "longitude" in ds_vis.coords:
        lon_var, lat_var = "longitude", "latitude"
    else:
        lon_var, lat_var = "lon", "lat"
    lons_vis = ds_vis[lon_var].values
    lats_vis = ds_vis[lat_var].values

    # 4) scatter‐plot
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        lons_vis, lats_vis,
        c=mean_da,
        s=10,
        edgecolor="none"
    )
    plt.colorbar(sc, label=cb_label)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Spatial Mean {var_label} (2015–2100)")
    plt.tight_layout()
    plt.show()

    ds_vis.close()

