import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# === 1. 读取黄土高原边界 Shapefile ===
shp_path = r'D:\EcoSci\Dr.Shi\Data\Loess_Plateau_vector_border\Loess_Plateau_vector_border\Loess_Plateau_vector_border.shp'
gdf = gpd.read_file(shp_path)

# 获取边界范围
minx, miny, maxx, maxy = gdf.total_bounds

# === 2. 生成 10 km × 10 km 空间网格 ===
grid_res_lat = 0.0898  # 10 km grid for latitude
grid_res_lon = 0.1084  # 10 km grid for longitude at 34°N

x_coords = np.arange(minx, maxx, grid_res_lon)  # Longitude grid
y_coords = np.arange(miny, maxy, grid_res_lat)  # Latitude grid
xx, yy = np.meshgrid(x_coords, y_coords)

# 创建网格 DataFrame
grid_df = pd.DataFrame({'LON': xx.ravel(), 'LAT': yy.ravel()})
grid_gdf = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.LON, grid_df.LAT), crs="EPSG:4326")

# === 3. 读取 CSV 采样数据 ===
csv_path = r'D:\EcoSci\Dr.Shi\Data\Loess_Plateau_Points.csv'
df = pd.read_csv(csv_path, encoding='ISO-8859-1')

# 保留原始列名
original_columns = df.columns.tolist()

# 确保经纬度是数值类型
df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # LON
df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # LAT

# 只选择数值列，排除文本列（如 LANDUSE）
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
points = df[num_cols[:2]].values  # 获取经纬度点

# === 4. 遍历数值列进行插值，并确保保留标题 ===
for col in num_cols[2:]:  # 仅对数值列插值
    values = df[col].values
    grid_values = griddata(points, values, (xx, yy), method='nearest', fill_value=np.nan)
    grid_df[col] = grid_values.ravel()

# === 5. 保存为 GeoDataFrame 并导出 ===
grid_gdf = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.LON, grid_df.LAT), crs="EPSG:4326")

# 仅保留落在黄土高原边界内的点
grid_gdf = grid_gdf[grid_gdf.geometry.within(gdf.union_all())]

# 确保只选择 `grid_gdf` 里存在的列，防止 KeyError
output_csv = r'D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_10km.csv'
valid_columns = [col for col in original_columns if col in grid_gdf.columns]
grid_gdf[valid_columns].to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"Resampled data saved to: {output_csv}")

# === 6. 可视化检查 ===
fig, ax = plt.subplots(figsize=(10, 8))
gdf.plot(ax=ax, edgecolor="black", facecolor="lightgray", alpha=0.5)
grid_gdf.plot(ax=ax, color='red', markersize=1, alpha=0.5)
ax.set_title("Resampled Data on 10km Grid in Loess Plateau")
plt.show()

