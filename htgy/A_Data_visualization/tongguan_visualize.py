import geopandas as gpd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# 潼关流域
tg_basin = gpd.read_file(DATA_DIR / "潼关以上流域.shp").to_crs("EPSG:4326")

# （如果有黄土高原shapefile，也可以加载）
loess_plateau = gpd.read_file(DATA_DIR / "Loess_Plateau_vector_border.shp").to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(10, 8))

tg_basin.boundary.plot(ax=ax, edgecolor='blue', linewidth=2, label="潼关以上流域")
loess_plateau.boundary.plot(ax=ax, edgecolor='green', linestyle='--', linewidth=1, label="黄土高原")

ax.set_title("潼关流域 vs 黄土高原范围")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.show()