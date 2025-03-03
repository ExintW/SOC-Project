import rasterio
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# File Paths
tiff_k1_path = r"D:\EcoSci\Dr.Shi\Data\k1_halfDegree.tif"
tiff_k2_path = r"D:\EcoSci\Dr.Shi\Data\k2_halfDegree.tif"
csv_file_path = r"D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_1km_with_DEM_region_labeled.csv"
output_csv_path = r"D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"

# Function to extract raster values at given coordinates
def extract_raster_values(tiff_path, lon, lat):
    with rasterio.open(tiff_path) as dataset:
        # Read raster data and transform
        raster_data = dataset.read(1)  # Read first band
        transform = dataset.transform
        nodata_value = dataset.nodata

        # Get raster pixel coordinates
        rows, cols = np.meshgrid(np.arange(raster_data.shape[0]), np.arange(raster_data.shape[1]), indexing="ij")

        # Convert pixel indices to geographic coordinates
        raster_lon, raster_lat = rasterio.transform.xy(transform, rows, cols)

        # Flatten arrays for interpolation
        raster_lon = np.array(raster_lon).flatten()
        raster_lat = np.array(raster_lat).flatten()
        raster_values = raster_data.flatten()

        # Handle NoData values by setting them to NaN
        if nodata_value is not None:
            mask = raster_values != nodata_value
            raster_lon, raster_lat, raster_values = raster_lon[mask], raster_lat[mask], raster_values[mask]

        # Interpolate raster values at given lon, lat
        extracted_values = griddata(
            (raster_lon, raster_lat), raster_values, (lon, lat), method="linear"
        )

        # If linear interpolation fails (NaN values remain), fall back to nearest neighbor
        nan_mask = np.isnan(extracted_values)
        if np.any(nan_mask):
            extracted_values[nan_mask] = griddata(
                (raster_lon, raster_lat), raster_values, (lon[nan_mask], lat[nan_mask]), method="nearest"
            )

        return extracted_values

# Function to convert SOM daily decomposition rate to SOC monthly decomposition rate
def convert_som_to_soc_monthly(k_day):
    """ Convert SOM daily rate (1/day) to SOC monthly rate (1/month). """
    k_month = 1 - np.exp(-k_day * 30)  # Convert daily to monthly
    return k_month * 0.58  # Apply SOC conversion factor

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Extract longitude and latitude
lon_csv = df["LON"].values
lat_csv = df["LAT"].values

# Extract k1 and k2 SOM values (1/day) from TIFFs
som_k1_day = extract_raster_values(tiff_k1_path, lon_csv, lat_csv)
som_k2_day = extract_raster_values(tiff_k2_path, lon_csv, lat_csv)

# Convert SOM to SOC monthly rates
df["SOC_k1_fast_pool (1/month)"] = convert_som_to_soc_monthly(som_k1_day)
df["SOC_k2_slow_pool (1/month)"] = convert_som_to_soc_monthly(som_k2_day)

# Save the updated CSV file
df.to_csv(output_csv_path, index=False)

print(f"✅ Updated CSV saved: {output_csv_path}")
