import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *
import xarray as xr
import pandas as pd
import numpy as np
# import geopandas as gpd  # Not needed since boundaries are not used
from pathlib import Path
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------------------
csv_path = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
ERA5_dir = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled"
# boarder_path = DATA_DIR / "Loess_Plateau_vector_border.shp"  # Boundary not used

# Output CSV file paths for each precipitation range
output_csv_range1 = PROCESSED_DIR / "k1_k2_60mm_pre_htgy.csv"
output_csv_range2 = PROCESSED_DIR / "k1_k2_90mm_pre_htgy.csv"
output_csv_range3 = PROCESSED_DIR / "k1_k2_120mm_pre_htgy.csv"

years = range(2007, 2025)
precip_var = "tp"
columns_to_keep = [
    "LAT", "LON", "Region",
    "SOC_k1_fast_pool (1/day)",
    "SOC_k2_slow_pool (1/day)",
    "SOC_k1_fast_pool (1/month)",
    "SOC_k2_slow_pool (1/month)",
    "TOTAL_PRECIP", "year", "month"
]

# ---------------------------------------------------------------------------------------
# Load the base CSV data without boundary filtering
# ---------------------------------------------------------------------------------------
df_base = pd.read_csv(csv_path)
print(f"Total number of base points: {len(df_base)}")

# ---------------------------------------------------------------------------------------
# Prepare containers for each precipitation range
# ---------------------------------------------------------------------------------------
upper_range_factor = 1.0001
lower_range_factor = 0.9999

results_range1 = []  # For range: 0.0594/30 to 0.0606/30
results_range2 = []  # For range: 0.0892/30 to 0.0908/30
results_range3 = []  # For range: 0.1188/30 to 0.1212/30

# ---------------------------------------------------------------------------------------
# Loop over each year and each month in the ERA5 datasets
# ---------------------------------------------------------------------------------------
for year in years:
    nc_file = ERA5_dir / f"resampled_{year}.nc"
    if not nc_file.exists():
        print(f"Warning: File not found: {nc_file}")
        continue

    ds = xr.open_dataset(nc_file)
    if precip_var not in ds.variables:
        print(f"Warning: Variable '{precip_var}' not found in {nc_file.name}. Skipping.")
        continue

    n_months = ds.sizes.get("valid_time", 0)
    if n_months == 0:
        print(f"Warning: No 'valid_time' dimension or dimension size=0 in {nc_file.name}. Skipping.")
        continue

    rename_dict = {}
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"
    if rename_dict:
        ds = ds.rename(rename_dict)

    lat_data = ds["lat"].values
    lon_data = ds["lon"].values
    print(f"Year {year}: lat shape {lat_data.shape}, lon shape {lon_data.shape}")

    points_data = np.column_stack((lat_data, lon_data))
    tree = cKDTree(points_data)
    query_points = np.column_stack((df_base["LAT"].values, df_base["LON"].values))

    for month_index in range(n_months):
        tp_data = ds[precip_var].isel(valid_time=month_index)
        tp_values = tp_data.values
        _, indices = tree.query(query_points)

        df_month = df_base.copy()
        df_month["TOTAL_PRECIP"] = tp_values[indices]

        converted_total = df_month["TOTAL_PRECIP"]
        mask1 = ((converted_total >= 0.06 * lower_range_factor/30) & (converted_total <= 0.06 * upper_range_factor/30))
        mask2 = ((converted_total >= 0.09 * lower_range_factor/30) & (converted_total <= 0.09 * upper_range_factor/30))
        mask3 = ((converted_total >= 0.12 * lower_range_factor/30) & (converted_total <= 0.12 * upper_range_factor/30))

        df_range1 = df_month[mask1].copy()
        if not df_range1.empty:
            df_range1["year"] = year
            df_range1["month"] = month_index + 1
            df_range1 = df_range1[columns_to_keep]
            results_range1.append(df_range1)

        df_range2 = df_month[mask2].copy()
        if not df_range2.empty:
            df_range2["year"] = year
            df_range2["month"] = month_index + 1
            df_range2 = df_range2[columns_to_keep]
            results_range2.append(df_range2)

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
    print(f"60mm pre data saved to: {output_csv_range1}")
else:
    print("No data processed for 60mm pre.")

if results_range2:
    final_df_range2 = pd.concat(results_range2, ignore_index=True)
    final_df_range2.to_csv(output_csv_range2, index=False)
    print(f"90mm pre data saved to: {output_csv_range2}")
else:
    print("No data processed for 90mm pre.")

if results_range3:
    final_df_range3 = pd.concat(results_range3, ignore_index=True)
    final_df_range3.to_csv(output_csv_range3, index=False)
    print(f"120mm pre data saved to: {output_csv_range3}")
else:
    print("No data processed for 120mm pre.")
