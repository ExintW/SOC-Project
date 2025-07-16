import os
import sys
from pathlib import Path

import xarray as xr
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# ─── adjust this to your project structure ─────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *
# ─────────────────────────────────────────────────────────────────────────────

# Define output directory for the maps
out_dir = OUTPUT_DIR / "Lai_Graphs"
out_dir.mkdir(parents=True, exist_ok=True)

# Path to Loess Plateau border shapefile
loess_border_path = Path(DATA_DIR) / "Loess_Plateau_vector_border.shp"

# File mappings
files = {
    "historical": "resampled_lai_points_1950-2000.nc",
    "present":    "resampled_lai_points_2001-2014.nc",
    "ssp126":     "resampled_lai_points_2015-2100_126.nc",
    "ssp245":     "resampled_lai_points_2015-2100_245.nc",
    "ssp370":     "resampled_lai_points_2015-2100_370.nc",
    "ssp585":     "resampled_lai_points_2015-2100_585.nc",
}

# Decadal years
hist_years   = list(range(1950, 2011, 10))  # 1950,60,…,2010
future_years = list(range(2020, 2101, 10))  # 2020,30,…,2100
scenarios    = ["ssp126", "ssp245", "ssp370", "ssp585"]

# Load Loess Plateau border
border = gpd.read_file(loess_border_path)
minx, miny, maxx, maxy = border.total_bounds

def plot_lai_map(ds, year, title, out_path):
    # average over the 12 months of that year
    lai_yr = ds['lai'].sel(time=slice(f"{year}-01-01", f"{year}-12-31")).mean(dim='time')

    # convert to GeoDataFrame
    lon = lai_yr['lon'].values
    lat = lai_yr['lat'].values
    vals = lai_yr.values
    df = pd.DataFrame({'lon': lon, 'lat': lat, 'lai': vals})
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=border.crs)

    # clip to Loess Plateau
    clipped = gpd.clip(gdf, border)

    # plot
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05

    fig, ax = plt.subplots(figsize=(16, 8))
    border.boundary.plot(ax=ax, linewidth=0.5, color='black')
    clipped.plot(column='lai', ax=ax, markersize=5, legend=True)

    # apply the padded limits
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Generate maps for 1950–2010
for year in hist_years:
    src = "historical" if year <= 2000 else "present"
    ds = xr.open_dataset(PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / files[src])
    out_fp = out_dir / f"lai_{year}.png"
    plot_lai_map(ds, year, f"LAI {year}", out_fp)
    ds.close()

# Generate decadal maps for each future scenario
for scen in scenarios:
    ds = xr.open_dataset(PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / files[scen])
    for year in future_years:
        out_fp = out_dir / f"lai_{scen}_{year}.png"
        plot_lai_map(ds, year, f"{scen.upper()} LAI {year}", out_fp)
    ds.close()

print(f"→ Saved all LAI maps to: {out_dir}")
