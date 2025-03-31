import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

# Adjust these paths according to your own directory structure
csv_path = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
ERA5_path = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / "resampled_2007.nc"

# Read CSV file into a pandas DataFrame
df = pd.read_csv(csv_path)

# Open the ERA5 NetCDF file using xarray
ds = xr.open_dataset(ERA5_path)

# Use the correct precipitation variable name; change if needed.
precip_var = "tp"

# Sum total precipitation over the time dimension (ensure "valid_time" is the correct dimension)
total_precip_ds = ds[precip_var].sum(dim="valid_time")

# Check the coordinate names of your dataset:
print("Dataset coordinates:", list(total_precip_ds.coords))

# Rename coordinates to 'lat' and 'lon' if needed
if "latitude" in total_precip_ds.coords and "longitude" in total_precip_ds.coords:
    total_precip_ds = total_precip_ds.rename({"latitude": "lat", "longitude": "lon"})
    print("Renamed 'latitude' and 'longitude' to 'lat' and 'lon'.")

# Print dimensions to see the structure (likely you'll see something like ('number',))
print("Dimensions:", total_precip_ds.dims)

# Since the data array is flat (e.g., with dimension "number"),
# extract the latitude and longitude coordinate arrays:
lat_data = total_precip_ds['lat'].values
lon_data = total_precip_ds['lon'].values

print("Shape of lat_data:", lat_data.shape, "Shape of lon_data:", lon_data.shape)

# Create a KDTree for the ERA5 grid points.
# We stack lat and lon to create a 2D array of points.
points_data = np.column_stack((lat_data, lon_data))
tree = cKDTree(points_data)

# For each coordinate from the CSV, build a 2D array (lat, lon)
query_points = np.column_stack((df['LAT'].values, df['LON'].values))

# Query the tree to get indices of the nearest ERA5 points
distances, indices = tree.query(query_points)

# Extract the precipitation values using these indices
sampled_values = total_precip_ds.values[indices]

# Add the sampled total precipitation values to the DataFrame
df["TOTAL_PRECIP"] = sampled_values

# Filter rows to keep only those whose total precipitation is in one of the specified intervals:
mask = (
    ((df["TOTAL_PRECIP"] >= 55) & (df["TOTAL_PRECIP"] <= 65)) |
    ((df["TOTAL_PRECIP"] >= 85) & (df["TOTAL_PRECIP"] <= 95)) |
    ((df["TOTAL_PRECIP"] >= 115) & (df["TOTAL_PRECIP"] <= 125))
)
df_filtered = df[mask]

# Save the filtered DataFrame to a CSV file
output_path = PROCESSED_DIR / "filtered_precip.csv"
df_filtered.to_csv(output_path, index=False)

print(f"Filtered rows saved to: {output_path}")
print(df_filtered.head())
