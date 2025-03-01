import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from shapely.geometry import Point

# 📝 设置支持中文和去除负号问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 📂 文件路径
shp_path = r"D:\EcoSci\Dr.Shi\Data\wammaogou_landuse\wammaogou_boundary.shp"  # 王茂沟边界
csv_path = r"D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_1km.csv"                  # 采样点
xls_path = r"D:\EcoSci\Dr.Shi\Data\wangmaogou_dams.xlsx"                      # 堤坝点位

# ✅ 读取王茂沟边界
gdf_boundary = gpd.read_file(shp_path)
boundary_polygon = gdf_boundary.geometry.union_all()  # 合并为单一多边形

# ✅ 读取采样点并转换为 GeoDataFrame
# ✅ 解决 BOM 问题
df_points = pd.read_csv(csv_path, encoding='utf-8-sig')

# 📝 检查 CSV 文件列名
print("📄 CSV 文件列名:", df_points.columns.tolist())

gdf_points = gpd.GeoDataFrame(
    df_points,
    geometry=gpd.points_from_xy(df_points["LON"], df_points["LAT"]),
    crs="EPSG:4326"
)

# ✅ 读取大坝点并转换为 GeoDataFrame
df_dams = pd.read_excel(xls_path)
gdf_dams = gpd.GeoDataFrame(
    df_dams,
    geometry=gpd.points_from_xy(df_dams["Longitude"], df_dams["Latitude"]),
    crs="EPSG:4326"
)

# ✅ 确保所有数据坐标系一致
gdf_points = gdf_points.to_crs(gdf_boundary.crs)
gdf_dams = gdf_dams.to_crs(gdf_boundary.crs)

# ✅ 提取边界内点位
points_within = gdf_points[gdf_points.within(boundary_polygon)]
dams_within = gdf_dams[gdf_dams.within(boundary_polygon)]

# 📊 绘图
fig, ax = plt.subplots(figsize=(12, 10))

# 绘制边界
gdf_boundary.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=2, label='王茂沟边界')

# 绘制采样点 (红色圆点)
if not points_within.empty:
    points_within.plot(ax=ax, color='red', marker='o', markersize=50, label='采样点位')
else:
    print("⚠️ 无采样点位位于边界内！")

# 绘制大坝点 (蓝色三角形)
dams_within.plot(ax=ax, color='blue', marker='^', markersize=70, label='大坝点位')

# 🎨 设置图例和标题
ax.set_title("王茂沟内的采样点与大坝点分布", fontsize=16, fontweight='bold')
ax.set_xlabel("经度", fontsize=12)
ax.set_ylabel("纬度", fontsize=12)
ax.legend(fontsize=12)

# ✅ 去除坐标轴科学计数法
ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
ax.ticklabel_format(style='plain', axis='x')  # X 轴普通格式
ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
ax.ticklabel_format(style='plain', axis='y')  # Y 轴普通格式

# ✅ 自适应坐标轴比例
ax.set_aspect('auto')

plt.show()

