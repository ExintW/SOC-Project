import pytesseract
from PIL import Image
import pandas as pd
import re
import sys
import os
import geopandas as gpd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# 定义多个shp文件路径（替换成你自己的文件路径）
shp_paths = [
    DATA_DIR / '潼关水文站.shp',
    DATA_DIR / '潼关以上流域.shp',
    DATA_DIR / '水系.shp',
    DATA_DIR / '干流.shp',
]

# 设置颜色样式（可选）
colors = ['red', 'green', 'darkblue', 'blue']
labels = ['TongGuan Station', 'Upper TongGuan Watershed', 'Drainage System', 'Main Stream']

# 创建画布
fig, ax = plt.subplots(figsize=(10, 8))

# 逐个读取并绘图
for path, color, label in zip(shp_paths, colors, labels):
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=4326)
    gdf.plot(ax=ax, edgecolor=color, facecolor='none', label=label)

# 添加图例和标题
ax.legend()
ax.set_title("多个矢量边界叠加图", fontsize=16)
plt.axis('equal')
plt.grid(True)

# 显示图形
plt.show()