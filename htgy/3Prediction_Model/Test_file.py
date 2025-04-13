
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # This should define PROCESSED_DIR and DATA_DIR

# Use the same paths as in your project
csv_path = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
boarder_path = DATA_DIR / "wammaogou_boundary.shp"

# Load the CSV data into a DataFrame
df = pd.read_csv(csv_path)

# Create a GeoDataFrame using the 'LON' and 'LAT' columns (assumed to be in EPSG:4326)
gdf_points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["LON"], df["LAT"]),
    crs="EPSG:4326"
)

# Load the WangMaoGou boundary shapefile
gdf_boundary = gpd.read_file(boarder_path)
# Reproject the boundary if its CRS differs from the points
if gdf_boundary.crs != gdf_points.crs:
    gdf_boundary = gdf_boundary.to_crs(gdf_points.crs)

# Combine all boundary geometries using union_all()
boundary = gdf_boundary.geometry.union_all()

# Filter the points that are within the boundary
points_within = gdf_points[gdf_points.within(boundary)]

# Print the number of points
print("Total points in CSV:", len(gdf_points))
print("Points within WangMaoGou boundary:", len(points_within))

# Optional: Plot the points and the boundary for visual inspection
ax = gdf_boundary.plot(edgecolor="red", facecolor="none", figsize=(8, 8))
gdf_points.plot(ax=ax, color="blue", markersize=5, label="All Points")
points_within.plot(ax=ax, color="green", markersize=5, label="Points within boundary")
plt.legend()
plt.title("Points within WangMaoGou Boundary")
plt.show()
