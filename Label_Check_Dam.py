import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# === 1. 读取数据 ===
shp_path = r'D:\EcoSci\Dr.Shi\Data\Loess_Plateau_vector_border\Loess_Plateau_vector_border\Loess_Plateau_vector_border.shp'
gdf = gpd.read_file(shp_path)

resampled_csv_path = r'D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_10km.csv'
grid_df = pd.read_csv(resampled_csv_path)

dam_xls_path = r"D:\EcoSci\Dr.Shi\Data\Dam_data.xlsx"
df_dams = pd.read_excel(dam_xls_path)

# === 2. 清理列名、确保数据格式 ===
grid_df.columns = grid_df.columns.str.strip()
df_dams.columns = df_dams.columns.str.strip()

grid_lon_col, grid_lat_col = "LON", "LAT"
dam_lon_col, dam_lat_col = "Longitude (°)", "Latitude (°)"

grid_df[grid_lon_col] = pd.to_numeric(grid_df[grid_lon_col], errors='coerce')
grid_df[grid_lat_col] = pd.to_numeric(grid_df[grid_lat_col], errors='coerce')
df_dams[dam_lon_col] = pd.to_numeric(df_dams[dam_lon_col], errors='coerce')
df_dams[dam_lat_col] = pd.to_numeric(df_dams[dam_lat_col], errors='coerce')

# === 3. 过滤 NaN / Inf 值 ===
df_dams = df_dams.dropna(subset=[dam_lon_col, dam_lat_col])
df_dams = df_dams[np.isfinite(df_dams[dam_lon_col]) & np.isfinite(df_dams[dam_lat_col])]

if df_dams.empty:
    raise ValueError("Error: No valid dam points found after filtering!")

# === 4. 计算每个堤坝点到最近的网格点（确保所有堤坝都匹配） ===
grid_points = np.vstack((grid_df[grid_lon_col], grid_df[grid_lat_col])).T
dam_points = np.vstack((df_dams[dam_lon_col], df_dams[dam_lat_col])).T

if dam_points.shape[0] == 0:
    raise ValueError("Error: No valid dam locations available for matching!")

# **使用 KD-Tree 查找最近邻**
tree = cKDTree(grid_points)
distances, nearest_grid_indices = tree.query(dam_points)

# === 5. 逐个匹配堤坝，确保所有堤坝点都唯一匹配一个网格点 ===
grid_df["check dam"] = 0  # 先初始化所有网格点为 0
matched_grid_points = set()  # 存储已匹配的网格点

for idx, grid_idx in enumerate(nearest_grid_indices):
    if grid_idx not in matched_grid_points:
        grid_df.at[grid_idx, "check dam"] = 1
        matched_grid_points.add(grid_idx)  # 标记该网格点已匹配
    else:
        print(f"堤坝点 {df_dams.iloc[idx][[dam_lon_col, dam_lat_col]].values} 被重复匹配到相同的网格点")

# === 6. 检查匹配结果 ===
matched_count = grid_df["check dam"].sum()
print(f"共有 {matched_count} 个网格点匹配了堤坝（目标: 86）")

unmatched_dams = len(df_dams) - matched_count
if unmatched_dams > 0:
    print(f"仍然有 {unmatched_dams} 个堤坝点未匹配到网格点！")

# === 7. 保存结果 ===
output_csv = r'D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_10km_with_dam.csv'
grid_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"Updated resampled data saved to: {output_csv}")
