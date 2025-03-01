import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# === 1. Load Loess Plateau Boundary Shapefile ===
shp_path = r'D:\EcoSci\Dr.Shi\Data\Loess_Plateau_vector_border\Loess_Plateau_vector_border\Loess_Plateau_vector_border.shp'
gdf = gpd.read_file(shp_path)

# Get boundary extent
minx, miny, maxx, maxy = gdf.total_bounds

# === 2. Generate 1 km × 1 km Grid ===
grid_res_lat = 0.00898  # ~1 km latitude
grid_res_lon = 0.01084  # ~1 km longitude at ~34°N

x_coords = np.arange(minx, maxx + grid_res_lon, grid_res_lon)
y_coords = np.arange(miny, maxy + grid_res_lat, grid_res_lat)
xx, yy = np.meshgrid(x_coords, y_coords)

# Create grid DataFrame
grid_df = pd.DataFrame({'LON': xx.ravel(), 'LAT': yy.ravel()})
grid_gdf = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.LON, grid_df.LAT), crs="EPSG:4326")

# === 3. Load Sample CSV Data ===
csv_path = r'D:\EcoSci\Dr.Shi\Data\Loess_Plateau_Points.csv'
df = pd.read_csv(csv_path, encoding='ISO-8859-1')

# Convert LON/LAT to numeric
df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # LON
df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # LAT

# Extract numeric columns and coordinates
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
points = df[num_cols[:2]].values  # Coordinates for interpolation

# === 4. Interpolate Numeric Columns ===
for col in num_cols[2:]:
    values = df[col].values
    grid_values = griddata(points, values, (xx, yy), method='nearest', fill_value=np.nan)
    grid_df[col] = grid_values.ravel()

# === 5. Resample 'LANDUSE' Using Nearest Neighbor (cKDTree) ===
if 'LANDUSE' in df.columns:
    # Build KDTree with sample points
    tree = cKDTree(points)

    # Query nearest point indices for each grid cell
    dist, idx = tree.query(grid_df[['LON', 'LAT']].values, k=1)

    # Assign LANDUSE from nearest sample point
    grid_df['LANDUSE'] = df['LANDUSE'].iloc[idx].values

# === 6. Keep Only Points Within the Loess Plateau Boundary ===
grid_gdf = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.LON, grid_df.LAT), crs="EPSG:4326")
grid_gdf = grid_gdf[grid_gdf.geometry.within(gdf.union_all())]

# === 7. Save to CSV ===
output_csv = r'D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_1km.csv'
grid_gdf.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"✅ Resampled data saved to: {output_csv}")

# === 8. Visualization ===
fig, ax = plt.subplots(figsize=(10, 8))
gdf.plot(ax=ax, edgecolor="black", facecolor="lightgray", alpha=0.5, label='Loess Plateau Boundary')
grid_gdf.plot(ax=ax, color='red', markersize=1, alpha=0.5, label='Resampled Grid Points')
ax.set_title("1 km × 1 km Resampled Data with Landuse (Loess Plateau)", fontsize=12)
ax.legend()
plt.show()
