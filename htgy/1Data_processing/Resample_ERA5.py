import os
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
working_dir = Path(__file__).parent.parent.parent
data_dir = working_dir / "Raw_Data"
processed_dir = working_dir / "Processed_Data"
ERA5_data_dir = data_dir / "ERA5"  # Directory containing .nc files
csv_file = processed_dir / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"  # CSV file path
output_dir = processed_dir / "ERA5_Data_Monthly_Resampled"  # Output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file containing longitude and latitude
df_points = pd.read_csv(csv_file)
lons = df_points["LON"].values  # Extract longitude
lats = df_points["LAT"].values  # Extract latitude

# Define the range of years
start_year = 1950
end_year = 2025

# Loop through each year and resample data
for year in range(start_year, end_year + 1):
    nc_file = os.path.join(ERA5_data_dir, f"{year}.nc")  # Construct file path

    # Check if the NetCDF file exists
    if not os.path.exists(nc_file):
        print(f"File {nc_file} not found, skipping...")
        continue

    # Open the NetCDF file
    try:
        ds = xr.open_dataset(nc_file)

        # Ensure correct variable names
        lon_name = "longitude" if "longitude" in ds.dims else "lon"
        lat_name = "latitude" if "latitude" in ds.dims else "lat"

        # Interpolate to match CSV file's lon/lat
        ds_resampled = ds.interp({lon_name: xr.DataArray(lons, dims="points"),
                                  lat_name: xr.DataArray(lats, dims="points")},
                                 method="nearest")

        # Save the resampled data
        output_file = os.path.join(output_dir, f"resampled_{year}.nc")
        ds_resampled.to_netcdf(output_file)

        # Print completion message
        print(f"✅ Resampling complete for {year}. Data saved to: {output_file}")

        # Close dataset
        ds.close()

    except Exception as e:
        print(f"❌ Error processing {year}: {e}")
