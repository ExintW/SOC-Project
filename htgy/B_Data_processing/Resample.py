import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# ========== 1. Read Loess Plateau Boundary Shapefile ==========
shp_path = DATA_DIR / "Loess_Plateau_vector_border.shp"
gdf_loess = gpd.read_file(shp_path)

# Get boundary extent
minx, miny, maxx, maxy = gdf_loess.total_bounds

# ========== 2. Generate 1 km × 1 km Grid ==========
# Approx 1 km in lat/lon degrees (varies with location, but close enough)
grid_res_lat = 0.00898  # ~1 km in latitude
grid_res_lon = 0.01084  # ~1 km in longitude (approx near 34°N)

# Generate coordinate arrays
x_coords = np.arange(minx, maxx + grid_res_lon, grid_res_lon)
y_coords = np.arange(miny, maxy + grid_res_lat, grid_res_lat)
xx, yy = np.meshgrid(x_coords, y_coords)

# Create a DataFrame for the grid
grid_df = pd.DataFrame({
    "LON": xx.ravel(),
    "LAT": yy.ravel()
})

# ========== 3. Read the Sample CSV (Loess Plateau Points) ==========
csv_path = DATA_DIR / "Loess_Plateau_Points.csv"
df = pd.read_csv(csv_path, encoding='ISO-8859-1')

# Ensure LON/LAT are numeric
df["LON"] = pd.to_numeric(df["LON"], errors='coerce')
df["LAT"] = pd.to_numeric(df["LAT"], errors='coerce')

# Drop any rows that are missing LON or LAT
df.dropna(subset=["LON", "LAT"], inplace=True)

# ========== 4. Build KDTree for Nearest-Neighbor Lookups ==========
points = df[["LON", "LAT"]].values  # coordinate array for sample points
tree = cKDTree(points)

# ========== 5. Interpolate Numeric Columns ==========
# Identify numeric columns, excluding the coordinate columns
# ========== 5. Interpolate Numeric Columns with Linear + Nearest‐Neighbor Fallback ==========
num_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in ["LON", "LAT"]
]

for col in num_cols:
    values = df[col].values

    # 5.1) Linear (bilinear on rectilinear grid) interpolation
    interp_linear = griddata(
        points, values,
        (xx, yy),
        method='linear',
        fill_value=np.nan
    )

    # 5.2) Nearest‐neighbor interpolation to cover hull/exterior
    interp_nearest = griddata(
        points, values,
        (xx, yy),
        method='nearest',
        fill_value=np.nan
    )

    # 5.3) Combine: use linear where available, otherwise nearest
    interp_combined = np.where(
        np.isnan(interp_linear),
        interp_nearest,
        interp_linear
    )

    # 5.4) Assign back into your grid DataFrame
    grid_df[col] = interp_combined.ravel()


# ========== 6. Resample 'LANDUSE' Using Nearest Neighbor ==========
if "LANDUSE" in df.columns:
    # For each grid cell, find the nearest sample point
    dist, idx = tree.query(grid_df[["LON", "LAT"]].values, k=1)
    # Assign the LANDUSE of the nearest sample point
    grid_df["LANDUSE"] = df["LANDUSE"].iloc[idx].values
else:
    print("Warning: 'LANDUSE' column not found in CSV.")

# ========== 7. Convert to GeoDataFrame and Clip to Loess Plateau ==========
grid_gdf = gpd.GeoDataFrame(
    grid_df,
    geometry=gpd.points_from_xy(grid_df["LON"], grid_df["LAT"]),
    crs="EPSG:4326"
)

# Clip to the Loess Plateau boundary (using unary_union of polygons)
grid_gdf = grid_gdf[grid_gdf.geometry.within(gdf_loess.union_all())]


# ========== 8. Save the Result to CSV ==========
output_csv = PROCESSED_DIR / "Resampled_Loess_Plateau_1km.csv"
grid_gdf.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"✅ Resampled data saved to: {output_csv}")

# ========== 9. Optional: Visualization ==========
fig, ax = plt.subplots(figsize=(10, 8))
gdf_loess.plot(ax=ax, edgecolor="black", facecolor="lightgray", alpha=0.5, label='Loess Plateau Boundary')
grid_gdf.plot(ax=ax, color='red', markersize=1, alpha=0.5, label='Resampled Grid Points')
ax.set_title("1 km × 1 km Resampled Data with Landuse (Loess Plateau)", fontsize=12)
ax.legend()
plt.show()

