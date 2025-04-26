import netCDF4 as nc
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import * 

def check_nc_lonlat_range(nc_file_path):
    """Check the lon/lat range of a NetCDF file."""
    with nc.Dataset(nc_file_path) as ds:
        lon = ds.variables['longitude'][:]
        lat = ds.variables['latitude'][:]
        print(f"Longitude range: {lon.min():.4f}°E ~ {lon.max():.4f}°E")
        print(f"Latitude range: {lat.min():.4f}°N ~ {lat.max():.4f}°N")
        print(f"Total lon points: {len(lon)}, Total lat points: {len(lat)}")

# Example usage
check_nc_lonlat_range(PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{2007}.nc")