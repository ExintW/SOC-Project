import os
import sys
import xarray as xr
import pandas as pd
from pathlib import Path

# Append project root to sys.path (if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # Assumes DATA_DIR and PROCESSED_DIR are defined here

# Define explicit file paths for LAI and PR
lai_file = DATA_DIR / "CMIP6" / "lai_Lmon_ACCESS-ESM1-5_ssp585_r1i1p1f1_gn_201501-210012.nc"
pr_file = DATA_DIR / "CMIP6" / "pr_Amon_ACCESS-ESM1-5_ssp585_r1i1p1f1_gn_201501-210012.hdf"

# Define output directory for resampled data
output_dir = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled"
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file containing target longitude and latitude points
csv_file = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
df_points = pd.read_csv(csv_file)
lons = df_points["LON"].values
lats = df_points["LAT"].values


def resample_dataset(ds, var_label, output_path):
    """
    Interpolates the input dataset spatially to the target points and saves the result as a NetCDF file.
    """
    # Determine coordinate names; CMIP6 datasets typically use "lon" and "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    lat_name = "latitude" if "latitude" in ds.dims else "lat"

    # Interpolate onto the provided points using nearest neighbor method
    ds_resampled = ds.interp({lon_name: xr.DataArray(lons, dims="points"),
                              lat_name: xr.DataArray(lats, dims="points")},
                             method="nearest")

    # Save the resampled dataset as a NetCDF file
    ds_resampled.to_netcdf(output_path)
    print(f"✅ Resampling complete for {var_label}. Data saved to: {output_path}")
    ds.close()


# Process LAI dataset (NetCDF)
try:
    ds_lai = xr.open_dataset(lai_file)
    output_file_lai = output_dir / "resampled_lai_2015-2100_585.nc"
    resample_dataset(ds_lai, "LAI", output_file_lai)
except Exception as e:
    print(f"❌ Error processing LAI: {e}")

# Process PR dataset (HDF -> NetCDF)
try:
    ds_pr = xr.open_dataset(pr_file)
    output_file_pr = output_dir / "resampled_pr_2015-2100_585.nc"
    resample_dataset(ds_pr, "Precipitation", output_file_pr)
except Exception as e:
    print(f"❌ Error processing PR: {e}")
