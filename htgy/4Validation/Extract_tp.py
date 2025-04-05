import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------------------
# CSV containing columns:
#   LAT, LON, Region, SOC_k1_fast_pool (1/day), SOC_k2_slow_pool (1/day),
#   SOC_k1_fast_pool (1/month), SOC_k2_slow_pool (1/month), etc.
csv_path = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"

# Directory containing the ERA5 NetCDF files named like resampled_2007.nc, resampled_2008.nc, ...
ERA5_dir = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled"

# Output CSV file paths for each precipitation range
output_csv_range1 = PROCESSED_DIR / "ERA5_extracted_range_0594_0606.csv"
output_csv_range2 = PROCESSED_DIR / "ERA5_extracted_range_0892_0908.csv"
output_csv_range3 = PROCESSED_DIR / "ERA5_extracted_range_1188_1212.csv"

# Years to process
years = range(2007, 2025)

# Name of the precipitation variable
precip_var = "tp"

# Columns to keep in the final output
columns_to_keep = [
    "LAT",
    "LON",
    "Region",
    "SOC_k1_fast_pool (1/day)",
    "SOC_k2_slow_pool (1/day)",
    "SOC_k1_fast_pool (1/month)",
    "SOC_k2_slow_pool (1/month)",
    "TOTAL_PRECIP",
    "year",
    "month"
]

# ---------------------------------------------------------------------------------------
# Load the base CSV data
# ---------------------------------------------------------------------------------------
df_base = pd.read_csv(csv_path)

# ---------------------------------------------------------------------------------------
# Prepare containers for each precipitation range
# ---------------------------------------------------------------------------------------
results_range1 = []  # For range: 0.0594/30 to 0.0606/30
results_range2 = []  # For range: 0.0892/30 to 0.0908/30
results_range3 = []  # For range: 0.1188/30 to 0.1212/30

# ---------------------------------------------------------------------------------------
# Loop over each year
# ---------------------------------------------------------------------------------------
for year in years:
    # Construct the path to the NetCDF file for this year
    nc_file = ERA5_dir / f"resampled_{year}.nc"

    if not nc_file.exists():
        print(f"Warning: File not found: {nc_file}")
        continue  # Skip this year if file is missing

    # Open the dataset
    ds = xr.open_dataset(nc_file)

    # Check variable presence
    if precip_var not in ds.variables:
        print(f"Warning: Variable '{precip_var}' not found in {nc_file.name}. Skipping.")
        continue

    # Use ds.sizes instead of ds.dims to avoid FutureWarnings
    n_months = ds.sizes.get("valid_time", 0)
    if n_months == 0:
        print(f"Warning: No 'valid_time' dimension or dimension size=0 in {nc_file.name}. Skipping.")
        continue

    # -----------------------------------------------------------------------------------
    # Check coordinate names and rename if necessary
    # -----------------------------------------------------------------------------------
    rename_dict = {}
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"

    if rename_dict:
        ds = ds.rename(rename_dict)

    # Build a KDTree using the ERA5 coordinate data (assumed to be 1D with dimension 'number')
    lat_data = ds["lat"].values
    lon_data = ds["lon"].values

    print(f"Year {year}: lat shape {lat_data.shape}, lon shape {lon_data.shape}")

    points_data = np.column_stack((lat_data, lon_data))
    tree = cKDTree(points_data)

    # Prepare query points from the CSV
    query_points = np.column_stack((df_base["LAT"].values, df_base["LON"].values))

    # -----------------------------------------------------------------------------------
    # Loop over each month in the current year
    # -----------------------------------------------------------------------------------
    for month_index in range(n_months):
        # Select the tp data for the current month
        tp_data = ds[precip_var].isel(valid_time=month_index)
        tp_values = tp_data.values  # should be a 1D array corresponding to the 'number' dimension

        # Find nearest neighbor indices for each query point
        _, indices = tree.query(query_points)

        # Create a copy of the base DataFrame so we can add columns
        df_month = df_base.copy()

        # Map the tp values using the nearest indices
        df_month["TOTAL_PRECIP"] = tp_values[indices]

        # In your original code the conversion is implied in the thresholds
        converted_total = df_month["TOTAL_PRECIP"]

        # Apply separate filtering masks for each range
        mask1 = ((converted_total >= 0.0594/30) & (converted_total <= 0.0606/30))
        mask2 = ((converted_total >= 0.0892/30) & (converted_total <= 0.0908/30))
        mask3 = ((converted_total >= 0.1188/30) & (converted_total <= 0.1212/30))

        # Process Range 1
        df_range1 = df_month[mask1].copy()
        if not df_range1.empty:
            df_range1["year"] = year
            df_range1["month"] = month_index + 1  # month_index is 0-based
            df_range1 = df_range1[columns_to_keep]
            results_range1.append(df_range1)

        # Process Range 2
        df_range2 = df_month[mask2].copy()
        if not df_range2.empty:
            df_range2["year"] = year
            df_range2["month"] = month_index + 1
            df_range2 = df_range2[columns_to_keep]
            results_range2.append(df_range2)

        # Process Range 3
        df_range3 = df_month[mask3].copy()
        if not df_range3.empty:
            df_range3["year"] = year
            df_range3["month"] = month_index + 1
            df_range3 = df_range3[columns_to_keep]
            results_range3.append(df_range3)

# ---------------------------------------------------------------------------------------
# Concatenate and save each range to separate CSV files
# ---------------------------------------------------------------------------------------
if results_range1:
    final_df_range1 = pd.concat(results_range1, ignore_index=True)
    final_df_range1.to_csv(output_csv_range1, index=False)
    print(f"Range 0.0594/30-0.0606/30 data saved to: {output_csv_range1}")
else:
    print("No data processed for range 0.0594/30-0.0606/30.")

if results_range2:
    final_df_range2 = pd.concat(results_range2, ignore_index=True)
    final_df_range2.to_csv(output_csv_range2, index=False)
    print(f"Range 0.0892/30-0.0908/30 data saved to: {output_csv_range2}")
else:
    print("No data processed for range 0.0892/30-0.0908/30.")

if results_range3:
    final_df_range3 = pd.concat(results_range3, ignore_index=True)
    final_df_range3.to_csv(output_csv_range3, index=False)
    print(f"Range 0.1188/30-0.1212/30 data saved to: {output_csv_range3}")
else:
    print("No data processed for range 0.1188/30-0.1212/30.")
