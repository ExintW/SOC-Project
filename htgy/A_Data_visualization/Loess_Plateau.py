import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# 读取矢量边界数据
shp_path = DATA_DIR / "Loess_Plateau_vector_border.shp"
gdf = gpd.read_file(shp_path)

# 读取原始点位（CSV 文件）
csv_path = DATA_DIR / 'Loess_Plateau_Points.csv'  # 替换为你的 CSV 文件路径
df_points = pd.read_csv(csv_path, encoding='ISO-8859-1')

# 读取堤坝点位（XLS 文件）
xls_path = DATA_DIR / "Dam_data.xlsx"  # 替换为你的 XLS 文件路径
df_dams = pd.read_excel(xls_path)

# 确保列名匹配（请确认 XLS 文件中的列名）
lon_col_points = "LON"
lat_col_points = "LAT"

lon_col_dams = "Longitude (°)"  # 假设堤坝文件也有相同的经纬度列
lat_col_dams = "Latitude (°)"

# 绘制地图
fig, ax = plt.subplots(figsize=(10, 8))
gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue")

# 绘制原始点位（红色 `o`）
ax.scatter(df_points[lon_col_points], df_points[lat_col_points], color='red', marker='o', s=40, label='General Points')

# 绘制堤坝点位（蓝色 `^`）
ax.scatter(df_dams[lon_col_dams], df_dams[lat_col_dams], color='blue', marker='^', s=60, label='Dam Points')

# 设置标题和图例
ax.set_title("Loess Plateau with General & Dam Points", fontsize=14)
ax.legend()

# 显示图形
plt.show()
