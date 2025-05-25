import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import csv
from pathlib import Path
import matplotlib.pyplot as plt   # ← added for plotting

# Append project root so we can import DATA_DIR, PROCESSED_DIR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# Input file path for CMIP6 LAI
lai_path = Path(DATA_DIR) / "CMIP6" / "lai_Lmon_BCC-CSM2-HR_hist-1950_r1i1p1f1_gn_200101-201412.nc"

# Load target 1 km grid points
csv_pts = Path(PROCESSED_DIR) / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
df_pts  = pd.read_csv(csv_pts)
lons    = df_pts["LON"].values
lats    = df_pts["LAT"].values

# Output directory for interpolated LAI data and stats
output_dir = Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled"
os.makedirs(output_dir, exist_ok=True)


def interp_and_collect(ds, var_name, label, save_interp_fname):
    """
    Interpolate ds[var_name] onto target points bilinearly,
    save the interpolated dataset, and return annual stats rows.
    """
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


# Process LAI only
lai_rows = []
try:
    ds_lai = xr.open_dataset(lai_path)
    print(">>> Processing LAI")
    lai_rows = interp_and_collect(
        ds_lai,
        var_name="lai",
        label="LAI",
        save_interp_fname="resampled_lai_points_2001-2014.nc"
    )
    ds_lai.close()
except Exception as e:
    print(f"Error processing LAI: {e}")

# Write annual stats to CSV
stats_file = output_dir / "annual_LAI_stats_2000-2014.csv"
with open(stats_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["variable", "year", "min", "max", "mean"])
    for row in lai_rows:
        writer.writerow(row)
print(f"→ Annual stats saved to {stats_file.name}")

# ────────────────────────────────────────────────────────────────────────────
# 4) Visualization: spatial map of mean LAI
# ────────────────────────────────────────────────────────────────────────────

# Path to the interpolated NetCDF we just created
interp_nc = output_dir / "resampled_lai_points_2001-2014.nc"
ds_res   = xr.open_dataset(interp_nc)

# Compute the long‐term mean over the time dimension
mean_lai = ds_res["lai"].mean(dim="time")
#year = 2012

# 2) Select all times in that year and average over them
#    Option A: using sel with a string slice
#mean_lai = ds_res["lai"].sel(time=f"{year}").mean(dim="time")

# Grab coordinates (detecting names in case they differ)
if "longitude" in ds_res.coords:
    lon_var, lat_var = "longitude", "latitude"
else:
    lon_var, lat_var = "lon", "lat"

lons = ds_res[lon_var].values
lats = ds_res[lat_var].values

# Plot
plt.figure(figsize=(10, 6))
sc = plt.scatter(
    lons, lats,
    c=mean_lai,
    s=10,
    edgecolor="none"
)
plt.colorbar(sc, label="Mean LAI (2000-2014)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of Mean LAI")
plt.tight_layout()
plt.show()

ds_res.close()