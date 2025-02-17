import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# === 1. 读取 Resampled 10km CSV 文件 ===
csv_path = r'D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_10km_with_dam_with_DEM.csv'
df = pd.read_csv(csv_path, encoding='utf-8-sig')

# 提取相关变量
lon_col, lat_col = "LON", "LAT"
soc_col = "ORGA"  # SOC 初始浓度
temp_col = "TEMP"  # 温度
moisture_col = "MOISTURE"  # 土壤湿度
ndvi_col = "NDVI"  # NDVI 植被指数
dem_col = "htgy_DEM"  # 地形高程
dam_col = "check dam"  # 淤堤坝信息 (1 = 有堤坝, 0 = 无堤坝)
rain_col = "RAIN"  # 降水量

# === 2. 转换为 2D 网格 ===
grid_x = np.sort(df[lon_col].unique())  # 经度
grid_y = np.sort(df[lat_col].unique())[::-1]  # 逆序，使纬度从大到小

grid_size = (len(grid_y), len(grid_x))  # 计算网格大小


def create_grid(data, col_name):
    return data.pivot(index=lat_col, columns=lon_col, values=col_name).sort_index(ascending=False).values


C = create_grid(df, soc_col)  # 初始 SOC 浓度 (g/kg)
NDVI = create_grid(df, ndvi_col)  # NDVI 值 (0-1)
NDVI = NDVI/10000 # 归一化
DEM = create_grid(df, dem_col)  # 高程数据
DAMS = create_grid(df, dam_col)  # 淤堤坝数据 (1: 有坝, 0: 无坝)
RAIN = create_grid(df, rain_col)  # 降水数据
SM = create_grid(df, moisture_col)  # 土壤湿度数据

# === 计算坡度 ===
Slope = np.gradient(DEM, axis=0) ** 2 + np.gradient(DEM, axis=1) ** 2
Slope = np.sqrt(Slope)


# === 计算侵蚀量 ===
def compute_erosion(RAIN, NDVI, Slope, SM, alpha=0.05, beta=1.5, gamma=10):
    return alpha *100*RAIN * (1 - NDVI) * (Slope ** beta) * (1 + gamma * SM)


E = compute_erosion(RAIN, NDVI, Slope, SM)


# === 计算坡度并分配沉积，包括淤堤坝影响 ===
def distribute_sediment(E, DEM, DAMS):
    D = np.zeros_like(E, dtype=float)
    for i in range(1, E.shape[0] - 1):
        for j in range(1, E.shape[1] - 1):
            neighbors = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                         (i, j - 1), (i, j + 1),
                         (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]

            if DAMS[i, j] == 1:
                for n in neighbors:
                    D[i, j] += E[n]
                    E[n] = 0
                continue

            valid_neighbors = []
            for n in neighbors:
                if DEM[n] < DEM[i, j]:
                    slope = (DEM[i, j] - DEM[n]) / np.sqrt((i - n[0]) ** 2 + (j - n[1]) ** 2)
                    valid_neighbors.append((n, slope))

            total_slope = sum(s for _, s in valid_neighbors)
            if total_slope > 0:
                for n, slope in valid_neighbors:
                    D[n] += (E[i, j] * slope) / total_slope
    return D


D = distribute_sediment(E, DEM, DAMS)


def vegetation_input(NDVI, base_input=1, max_input= 1000):
    return max_input * (1 - np.exp(-NDVI / base_input))


V = vegetation_input(NDVI)


def soc_dynamic_model(C_fast, C_slow, E, D, V, T, SM, NDVI, k_base, dx, dt, p_fast=0.5):
    k_fast = k_base * 10 * (1 + 0.1 * T) * (1 + 0.05 * SM) * (1 - NDVI)
    k_slow = k_base * 0.1 * (1 + 0.1 * T) * (1 + 0.05 * SM) * (1 - NDVI)

    erosion_fast = -E * C_fast
    erosion_slow = -E * C_slow

    deposition_fast = D * p_fast
    deposition_slow = D * (1 - p_fast)

    vegetation_fast = V * p_fast
    vegetation_slow = V * (1 - p_fast)

    reaction_fast = -k_fast * C_fast
    reaction_slow = -k_slow * C_slow

    C_fast_new = C_fast + (erosion_fast + deposition_fast + vegetation_fast + reaction_fast) * dt
    C_slow_new = C_slow + (erosion_slow + deposition_slow + vegetation_slow + reaction_slow) * dt

    return np.maximum(C_fast_new, 0), np.maximum(C_slow_new, 0)


k_base = 0.001
dx = 10
dt = 1
timesteps = 50

p_fast = 0.5
C_fast = C * p_fast
C_slow = C * (1 - p_fast)

T_series = [create_grid(df, temp_col) + np.random.randn(*grid_size) * 1.5 for _ in range(timesteps)]
SM_series = [create_grid(df, moisture_col) + np.random.randn(*grid_size) * 5 for _ in range(timesteps)]

for t in range(timesteps):
    if t > 0:
        T = T_series[t]
        SM = SM_series[t]
        C_fast, C_slow = soc_dynamic_model(C_fast, C_slow, E, D, V, T, SM, NDVI, k_base, dx, dt, p_fast)

    if t %10 == 0:
        plt.imshow(C_fast + C_slow, cmap="viridis", extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        plt.colorbar(label="SOC Concentration (g/kg)")
        plt.title(f"SOC Concentration at Year {t}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()


