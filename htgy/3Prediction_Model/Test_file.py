import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from pathlib import Path
import sys

# Append globals (assumes DATA_DIR, PROCESSED_DIR, OUTPUT_DIR are defined in globals.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# ---------------------------
# Read region CSV to obtain grid coordinates.
# ---------------------------
region_csv_path = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
df = gpd.read_file(region_csv_path)  # Alternatively, use pd.read_csv if it's a CSV

# Define column names.
lon_col, lat_col = "LON", "LAT"

# Ensure coordinate columns are numeric.
df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')

# Extract unique grid coordinates.
grid_x = np.sort(df[lon_col].unique())
grid_y = np.sort(df[lat_col].unique())[::-1]  # descending order for latitude

# For demonstration, create a dummy DEM using grid dimensions.
N = len(grid_y)
M = len(grid_x)
DEM = np.tile(np.linspace(100, 0, N).reshape(N, 1), (1, M))

# =============================================================================
# BUFFERING THE SHAPEFILES AND CREATING MASK ARRAYS
# =============================================================================
# Define shapefile paths.
small_boundary_shp_path = DATA_DIR / "River_Basin" / "htgy_River_Basin.shp"
large_boundary_shp_path = DATA_DIR / "River_Basin" / "94_area.shp"
river_shp_path = DATA_DIR / "China_River" / "ChinaRiver_main.shp"

# Read the shapefiles.
small_boundary_shp = gpd.read_file(small_boundary_shp_path)
large_boundary_shp = gpd.read_file(large_boundary_shp_path)
river_shp = gpd.read_file(river_shp_path)

# Compute grid resolution and define buffer distance.
dx = np.mean(np.diff(grid_x))
dy = np.mean(np.diff(grid_y))
resolution = np.mean([dx, dy])
buffer_distance = resolution / 2  # Buffer half the cell size

# Buffer the shapefiles to create narrow polygons.
small_boundary_buffered = small_boundary_shp.buffer(buffer_distance)
large_boundary_buffered = large_boundary_shp.buffer(buffer_distance)
river_buffered = river_shp.buffer(buffer_distance)

# Combine buffered geometries using union_all() (avoiding deprecated unary_union).
small_boundary_union = small_boundary_buffered.union_all()
large_boundary_union = large_boundary_buffered.union_all()
river_union = river_buffered.union_all()

# Create a meshgrid of cell centers.
X, Y = np.meshgrid(grid_x, grid_y)

# Define a helper function for point-in-polygon test.
def point_intersects(geom, x, y):
    return geom.intersects(Point(x, y))

# Use np.vectorize to create boolean mask arrays.
small_boundary_mask = np.vectorize(lambda x, y: point_intersects(small_boundary_union, x, y))(X, Y)
large_boundary_mask = np.vectorize(lambda x, y: point_intersects(large_boundary_union, x, y))(X, Y)
river_mask = np.vectorize(lambda x, y: point_intersects(river_union, x, y))(X, Y)

# =============================================================================
# PLOTTING THE OVERLAY FOR VISUAL INSPECTION
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 10))
# Plot DEM as the background.
im = ax.imshow(DEM, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
               cmap='terrain', origin='upper')
plt.colorbar(im, ax=ax, label='Elevation')

# Overlay the buffered boundaries and river.
small_boundary_buffered.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, label='Small Basin Boundary')
large_boundary_buffered.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=2, label='Large Basin Boundary')
river_buffered.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, label='Main River')

ax.legend()
ax.set_title("Overlay of Buffered Boundaries and Main River")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()
