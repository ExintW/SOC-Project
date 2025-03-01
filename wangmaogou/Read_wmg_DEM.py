import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point

# === 1. 设置文件路径 ===
csv_file = r"D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_1km.csv"  # 输入的点位CSV
shp_path = r"D:\EcoSci\Dr.Shi\Data\wammaogou_landuse\wammaogou_boundary.shp"  # 王茂沟边界
adf_folder = r"D:\EcoSci\Dr.Shi\Data\DEM-wanmaogou\DEM\wmg\hdr.adf"  # ADF文件路径
output_csv = r"D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_wangmaogou_1km_with_DEM.csv"  # 输出CSV路径

# === 2. 读取数据 ===
df = pd.read_csv(csv_file)
boundary = gpd.read_file(shp_path)

# 检查 CSV 是否包含所需列
if "LON" not in df.columns or "LAT" not in df.columns:
    raise ValueError("CSV 文件中缺少 'LON' 或 'LAT' 列")

# === 3. 检查 CRS 和空间范围 ===
with rasterio.open(adf_folder) as dem_dataset:
    dem_crs = dem_dataset.crs
    dem_bounds = dem_dataset.bounds
    print(f"DEM CRS: {dem_crs}")
    print(f"DEM Bounds: {dem_bounds}")

    print(f"Boundary CRS: {boundary.crs}")
    print(f"Boundary Bounds: {boundary.total_bounds}")

    # 如果 CRS 不匹配，进行重投影
    if boundary.crs != dem_crs:
        print("CRS 不匹配，正在重投影边界以匹配 DEM 的 CRS...")
        boundary = boundary.to_crs(dem_crs)
        print(f"重投影后的 Boundary Bounds: {boundary.total_bounds}")
    else:
        print("CRS 已匹配！")

# === 4. 检查边界与 DEM 是否重叠 ===
boundary_bounds = boundary.total_bounds
dem_extent = (dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)

if not (
    (boundary_bounds[2] >= dem_extent[0]) and
    (boundary_bounds[0] <= dem_extent[2]) and
    (boundary_bounds[3] >= dem_extent[1]) and
    (boundary_bounds[1] <= dem_extent[3])
):
    raise ValueError("边界与 DEM 不重叠！请检查坐标系和地理位置。")

# === 5. 提取边界内点位 ===
geometry = [Point(lon, lat) for lon, lat in zip(df["LON"], df["LAT"])]
points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # 假设CSV是经纬度
points_gdf = points_gdf.to_crs(dem_crs)  # 转换为DEM坐标系

points_within_boundary = gpd.sjoin(points_gdf, boundary, how="inner", predicate='within')
if points_within_boundary.empty:
    raise ValueError("没有点位位于王茂沟边界内！")

# === 6. 读取 DEM 并提取 DEM 值 ===
with rasterio.open(adf_folder) as dem_dataset:
    # 使用裁剪功能确保 DEM 范围正确
    out_image, out_transform = mask(dem_dataset, boundary.geometry, crop=True)
    out_image = out_image[0]

    dem_values = []
    for point in points_within_boundary.geometry:
        row, col = dem_dataset.index(point.x, point.y)
        if 0 <= row < dem_dataset.height and 0 <= col < dem_dataset.width:
            dem_value = out_image[row, col]
            dem_value = None if dem_dataset.nodata and dem_value == dem_dataset.nodata else dem_value
        else:
            dem_value = None
        dem_values.append(dem_value)

# === 7. 保存结果 ===
points_within_boundary["htgy_DEM"] = dem_values
points_within_boundary.drop(columns="geometry", inplace=True)
points_within_boundary.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"处理完成！新 CSV 文件已保存至: {output_csv}")
