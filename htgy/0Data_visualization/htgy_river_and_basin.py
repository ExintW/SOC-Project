import geopandas as gpd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# Read the Loess Plateau and sub-basins boundary
shapefile_path_boundary = DATA_DIR / "River_Basin" / "htgy_River_Basin.shp"
gdf_boundary = gpd.read_file(shapefile_path_boundary)

# Read the additional boundary shapefile
shapefile_path_additional = DATA_DIR / "River_Basin" / "94_area.shp"
gdf_additional_boundary = gpd.read_file(shapefile_path_additional)

# Reproject the additional boundary to match the main boundary's CRS if needed
if gdf_additional_boundary.crs != gdf_boundary.crs:
    gdf_additional_boundary = gdf_additional_boundary.to_crs(gdf_boundary.crs)

# Read the smallest river data
shapefile_path_river = DATA_DIR / "China_River" / "China_River_smallest.shp"
gdf_river = gpd.read_file(shapefile_path_river)

# Reproject the river data if its CRS does not match the boundary
if gdf_river.crs != gdf_boundary.crs:
    gdf_river = gdf_river.to_crs(gdf_boundary.crs)

# Clip the river data to the Loess Plateau boundary
gdf_river_clipped = gpd.overlay(gdf_river, gdf_boundary, how="intersection")

# Read the main river data and process it similarly
shapefile_path_river_main = DATA_DIR / "China_River" / "ChinaRiver_main.shp"
gdf_river_main = gpd.read_file(shapefile_path_river_main)

# Reproject the main river data if necessary
if gdf_river_main.crs != gdf_boundary.crs:
    gdf_river_main = gdf_river_main.to_crs(gdf_boundary.crs)

# Clip the main river data to the Loess Plateau boundary
gdf_river_main_clipped = gpd.overlay(gdf_river_main, gdf_boundary, how="intersection")

# Create a figure and axis for plotting
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the Loess Plateau and sub-basins boundary
gdf_boundary.plot(
    ax=ax,
    edgecolor="black",
    facecolor="none",
    linewidth=0.5,
    label="River Basin"
)

# Plot the additional boundary with a dashed green outline
gdf_additional_boundary.plot(
    ax=ax,
    edgecolor="green",
    facecolor="none",
    linewidth=1,
    linestyle="--",
    label="94 area"
)

# Plot the clipped smallest river data in blue
gdf_river_clipped.plot(
    ax=ax,
    color="blue",
    linewidth=0.5,
    label="River"
)

# Plot the clipped main river data in red
gdf_river_main_clipped.plot(
    ax=ax,
    color="red",
    linewidth=0.5,
    label="Main River"
)

ax.set_title("Loess Plateau River Basin", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True)
plt.legend()
plt.show()
