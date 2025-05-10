import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Point
import numpy as np
from scipy.spatial import cKDTree
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# === 1. 设置文件路径 ===
csv_file = PROCESSED_DIR / "Resampled_Loess_Plateau_1km.csv"  # 输入的点位CSV
shp_path = DATA_DIR / "Loess_Plateau_vector_border.shp"  # htgy边界
tif_path = DATA_DIR / "htgyDEM.tif"  # TIFF文件路径
output_csv = PROCESSED_DIR / "Resampled_Loess_Plateau_1km_with_DEM.csv"  # 输出CSV路径

# === 2. 读取数据 ===
df = pd.read_csv(csv_file)
boundary = gpd.read_file(shp_path)

# 检查 CSV 是否包含所需列
if "LON" not in df.columns or "LAT" not in df.columns:
    raise ValueError("CSV 文件中缺少 'LON' 或 'LAT' 列")

# === 3. 读取 DEM 数据，检查 CRS 和空间范围 ===
with rasterio.open(tif_path) as dem_dataset:
    dem_crs = dem_dataset.crs
    dem_bounds = dem_dataset.bounds
    print(f"DEM CRS: {dem_crs}")
    print(f"DEM Bounds: {dem_bounds}")

    # 检查边界文件 CRS 并进行转换
    if boundary.crs != dem_crs:
        print("CRS 不匹配，正在重投影边界...")
        boundary = boundary.to_crs(dem_crs)

# === 4. 处理点数据，确保其在 DEM 范围内 ===
geometry = [Point(lon, lat) for lon, lat in zip(df["LON"], df["LAT"])]
points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # 假设 CSV 是 WGS84 经纬度
points_gdf = points_gdf.to_crs(dem_crs)  # 转换为 DEM 坐标系

# === 5. 筛选出边界范围内的点 ===
points_within_boundary = gpd.sjoin(points_gdf, boundary, how="inner", predicate='within')
if points_within_boundary.empty:
    raise ValueError("没有点位位于 Loess Plateau 边界内！")

# === 6. 提取 DEM 高程值（修复 sample() 调用）===
dem_values = []
with rasterio.open(tif_path) as dem_dataset:
    for point in points_within_boundary.geometry:
        try:
            value = next(dem_dataset.sample([(point.x, point.y)]))  # 直接使用 sample() 方法
            dem_value = value[0] if value is not None else None
        except StopIteration:
            dem_value = None
        dem_values.append(dem_value)

points_within_boundary["htgy_DEM"] = dem_values

# === 7. 处理缺失值（使用最近邻插值填充）===
if points_within_boundary["htgy_DEM"].isnull().sum() > 0:
    print("发现 DEM 值缺失点，正在进行最近邻插值填充...")

    # 获取有效DEM值的点
    valid_points = points_within_boundary.dropna(subset=["htgy_DEM"])
    valid_coords = np.array([(p.x, p.y) for p in valid_points.geometry])
    valid_values = np.array(valid_points["htgy_DEM"])

    # 构建 KD-Tree 进行最近邻搜索
    tree = cKDTree(valid_coords)

    # 查找最近邻的非空值
    missing_points = points_within_boundary[points_within_boundary["htgy_DEM"].isnull()]
    missing_coords = np.array([(p.x, p.y) for p in missing_points.geometry])
    _, idx = tree.query(missing_coords, k=1)

    # 用最近邻的值填充缺失值
    points_within_boundary.loc[points_within_boundary["htgy_DEM"].isnull(), "htgy_DEM"] = valid_values[idx]


# 确保 boundary 和 points_within_boundary 在同一 CRS（上面已投影到 dem_crs）
fig, ax = plt.subplots(figsize=(10, 8))

# 画出边界
boundary.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1)

# 按照 htgy_DEM 值给点着色
points_within_boundary.plot(
    ax=ax,
    column='htgy_DEM',
    cmap='terrain',
    markersize=5,
    legend=True,
    legend_kwds={'label': 'Elevation (m)', 'shrink': 0.6}
)

ax.set_title("Resampled DEM Values on Loess Plateau Grid Points")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
plt.tight_layout()
plt.show()

# === 8. 保存结果 ===
points_within_boundary.drop(columns="geometry", inplace=True)
points_within_boundary.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"处理完成！新 CSV 文件已保存至: {output_csv}")
