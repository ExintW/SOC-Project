import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# === 1. 读取数据 ===
shp_path = r"D:\EcoSci\Dr.Shi\Data\Loess_Plateau_vector_border\Loess_Plateau_vector_border\Loess_Plateau_vector_border.shp"
gdf = gpd.read_file(shp_path)

resampled_csv_path = r"D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_1km_with_DEM.csv"
grid_df = pd.read_csv(resampled_csv_path)

dam_csv_path = r"D:\EcoSci\Dr.Shi\Data\htgy_dams_fixed.csv"
df_dams = pd.read_csv(dam_csv_path)

# === 2. 清理列名、确保数据格式 ===
grid_df.columns = grid_df.columns.str.strip()
df_dams.columns = df_dams.columns.str.strip()

grid_lon_col, grid_lat_col = "LON", "LAT"
dam_lon_col, dam_lat_col = "x", "y"

grid_df[grid_lon_col] = pd.to_numeric(grid_df[grid_lon_col], errors='coerce')
grid_df[grid_lat_col] = pd.to_numeric(grid_df[grid_lat_col], errors='coerce')
df_dams[dam_lon_col] = pd.to_numeric(df_dams[dam_lon_col], errors='coerce')
df_dams[dam_lat_col] = pd.to_numeric(df_dams[dam_lat_col], errors='coerce')

# === 3. 过滤 NaN / Inf 值 ===
df_dams = df_dams.dropna(subset=[dam_lon_col, dam_lat_col])
df_dams = df_dams[np.isfinite(df_dams[dam_lon_col]) & np.isfinite(df_dams[dam_lat_col])]

if df_dams.empty:
    raise ValueError("Error: No valid dam points found after filtering!")

# === 4. KD-Tree 查找最近网格点 ===
grid_points = np.vstack((grid_df[grid_lon_col], grid_df[grid_lat_col])).T
dam_points = np.vstack((df_dams[dam_lon_col], df_dams[dam_lat_col])).T

tree = cKDTree(grid_points)
distances, nearest_grid_indices = tree.query(dam_points)

# === 5. 将匹配结果添加到堤坝 DataFrame ===
df_dams["matched_Lon"] = grid_df.loc[nearest_grid_indices, grid_lon_col].values
df_dams["matched_Lat"] = grid_df.loc[nearest_grid_indices, grid_lat_col].values

# === 6. 标记网格点是否为沉积区 ===
grid_df["Region"] = "erosion area"  # 默认值
matched_indices = set(nearest_grid_indices)
grid_df.loc[grid_df.index.isin(matched_indices), "Region"] = "sedimentation area"

# === 7. 保存更新后的文件 ===

# 7.1 保存更新后的堤坝数据到 CSV
dam_output_csv = r"D:\EcoSci\Dr.Shi\Data\htgy_Dam_with_matched_points.csv"
df_dams.to_csv(dam_output_csv, index=False, encoding='utf-8-sig')
print(f"堤坝匹配结果已保存至: {dam_output_csv}")

# 7.2 保存更新后的网格数据带有 Region 列
grid_output_csv = r"D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_1km_with_DEM_region_labeled.csv"
grid_df.to_csv(grid_output_csv, index=False, encoding='utf-8-sig')
print(f"网格数据已更新并保存至: {grid_output_csv}")

# === 8. 匹配结果统计 ===
matched_count = df_dams["matched_Lon"].notna().sum()
print(f"成功为 {matched_count} 个堤坝匹配到最近网格点。")
sedimentation_count = (grid_df["Region"] == "sedimentation area").sum()
erosion_count = (grid_df["Region"] == "erosion area").sum()

print(f"沉积区数量: {sedimentation_count}, 侵蚀区数量: {erosion_count}")
