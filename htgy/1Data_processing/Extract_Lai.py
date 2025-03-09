import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import re
from pathlib import Path

# File paths
working_dir = Path(__file__).parent.parent.parent
data_dir = working_dir / "Raw_Data"
output_dir = working_dir / "Processed_Data"
nc_file = data_dir / "2007.nc"
csv_file = data_dir / "Vegetation_Input_v2.csv"
output_csv = output_dir / "Vegetation_Input_v2_with_Lai.csv"

# Load the NetCDF file
ds = xr.open_dataset(nc_file)

# Compute the mean LAI for all months in 2007
lai_data = ds['lai_lv'].mean(dim='valid_time', skipna=True)

# Load CSV file
df = pd.read_csv(csv_file, dtype=str)  # Read everything as string for cleaning
print("Original CSV Column Names:", df.columns)

# Standardize column names
df.columns = df.columns.str.strip()

# Identify correct latitude and longitude columns
lat_col = 'Latitude'
lon_col = 'Longitude'

# Function to clean numeric values (removing spaces, special characters)
def clean_numeric(value):
    if isinstance(value, str):
        value = re.sub(r'[^\d\.-]', '', value)  # Keep only numbers, dots, and minus signs
        try:
            return float(value) if value else None
        except ValueError:
            return None
    return value

# Clean and convert latitude and longitude
df[lat_col] = df[lat_col].apply(clean_numeric)
df[lon_col] = df[lon_col].apply(clean_numeric)

# Debugging: Print rows with missing Latitude/Longitude
missing_coords = df[df[lat_col].isna() | df[lon_col].isna()]
if not missing_coords.empty:
    print("Rows removed due to invalid coordinates:\n", missing_coords)

# Drop rows where Latitude or Longitude is missing
df = df.dropna(subset=[lat_col, lon_col])

# Extract lat/lon from NetCDF
nc_lats = ds['latitude'].values
nc_lons = ds['longitude'].values

# Create a KDTree for fast nearest neighbor search
grid_points = np.array([(lat, lon) for lat in nc_lats for lon in nc_lons])
tree = cKDTree(grid_points)

# Function to get the nearest average LAI value
def get_nearest_lai(lat, lon):
    if np.isnan(lat) or np.isnan(lon):
        return np.nan
    _, idx = tree.query([lat, lon])
    nearest_lat, nearest_lon = grid_points[idx]

    try:
        lai_value = lai_data.sel(latitude=nearest_lat, longitude=nearest_lon, method='nearest').values.item()
        print(f"Matching ({lat}, {lon}) -> ({nearest_lat}, {nearest_lon}) | LAI: {lai_value}")  # Debug print
        return lai_value
    except Exception as e:
        print(f"Error finding LAI for ({lat}, {lon}): {e}")
        return np.nan

# Apply function to extract the yearly average LAI
df['Lai'] = df.apply(lambda row: get_nearest_lai(row[lat_col], row[lon_col]), axis=1)

# Save updated CSV with UTF-8 encoding to fix character issues
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"Updated CSV saved: {output_csv}")

