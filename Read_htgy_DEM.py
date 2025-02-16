import rasterio
import pandas as pd
import numpy as np
from rasterio.transform import rowcol

# 设置文件路径
csv_file = r"D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_10km_with_dam.csv"
tiff_file = r"D:\EcoSci\Dr.Shi\Data\htgyDEM-30m\htgyDEM.tif"
output_csv = r"D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_10km_with_dam_with_DEM.csv"

# 读取 CSV 文件
df = pd.read_csv(csv_file)

# 确保 CSV 文件包含 'LON' 和 'LAT' 列
if "LON" not in df.columns or "LAT" not in df.columns:
    raise ValueError("CSV 文件中缺少 'LON' 或 'LAT' 列")

# 打开 TIFF 文件
with rasterio.open(tiff_file) as dataset:
    transform = dataset.transform  # 获取仿射变换
    dem_array = dataset.read(1)  # **一次性读取整个 DEM 数组，提高速度**

    dem_values = []  # 存储每个点的 DEM 值

    for lon, lat in zip(df["LON"], df["LAT"]):
        # 将地理坐标 (lon, lat) 转换为像素索引 (row, col)
        row, col = rowcol(transform, lon, lat)

        # 确保点位在 DEM 数据范围内
        if 0 <= row < dataset.height and 0 <= col < dataset.width:
            dem_value = dem_array[row, col]  # 直接从 numpy 数组中获取数据（**更快！**）
        else:
            dem_value = None  # 处理超出范围的点

        dem_values.append(dem_value)

# 添加新列
df["htgy_DEM"] = dem_values

# 保存结果
df.to_csv(output_csv, index=False)

print(f"处理完成！带有 DEM 数据的新 CSV 文件已保存至: {output_csv}")

