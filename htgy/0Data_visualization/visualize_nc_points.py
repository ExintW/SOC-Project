import netCDF4 as nc
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import * 

def plot_nc_lonlat_density(nc_file_path, save_fig=False):
    """Plot the distribution of lon/lat points from a NetCDF file."""
    with nc.Dataset(nc_file_path) as ds:
        lon = ds.variables['longitude'][:]
        lat = ds.variables['latitude'][:]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(lon, lat, s=1, alpha=0.5)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Distribution of Points in NetCDF")
    ax.grid(True)
    
    if save_fig:
        fig.savefig("lonlat_distribution.png", dpi=300)
        print("Saved figure as lonlat_distribution.png")
    plt.show()

# Example usage
plot_nc_lonlat_density(PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{2007}.nc", save_fig=False)