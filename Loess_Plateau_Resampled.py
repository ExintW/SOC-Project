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

# === 2. 转换为 2D 网格 ===
grid_x = np.sort(df[lon_col].unique())  # 经度
grid_y = np.sort(df[lat_col].unique())[::-1]  # 逆序，使纬度从大到小

grid_size = (len(grid_y), len(grid_x))  # 计算网格大小

def create_grid(data, col_name):
    return data.pivot(index=lat_col, columns=lon_col, values=col_name).sort_index(ascending=False).values

C = create_grid(df, soc_col)  # 初始 SOC 浓度 (g/kg)
E = np.random.rand(*grid_size) * 0.05  # 侵蚀速率 (t/ha/year)
NDVI = create_grid(df, ndvi_col)  # NDVI 值 (0-1)
DEM = create_grid(df, dem_col)  # 高程数据
DAMS = create_grid(df, dam_col)  # 淤堤坝数据 (1: 有坝, 0: 无坝)

# === 计算坡度并分配沉积，包括淤堤坝影响 ===
def distribute_sediment(E, DEM, DAMS):
    """
    计算坡度并按比例分配沉积 SOC，考虑淤堤坝的影响：
    - 如果一个网格有坝，则所有邻居的侵蚀 SOC 强制沉积到该网格。
    - 如果没有坝，按照坡度比例正常沉积。
    """
    D = np.zeros_like(E, dtype=float)

    for i in range(1, E.shape[0] - 1):
        for j in range(1, E.shape[1] - 1):
            # 获取 8 邻域网格
            neighbors = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                         (i, j - 1), (i, j + 1),
                         (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]

            # 如果当前网格有坝，所有邻居的 SOC 都沉积到这里
            if DAMS[i, j] == 1:
                for n in neighbors:
                    D[i, j] += E[n]  # 将邻居的侵蚀 SOC 全部沉积到当前网格
                    E[n] = 0  # 邻居的侵蚀量清零
                continue  # 直接跳过坡度计算

            # 正常坡度分配（只有无坝网格才执行）
            valid_neighbors = []
            for n in neighbors:
                if DEM[n] < DEM[i, j]:  # 只往低处流动
                    slope = (DEM[i, j] - DEM[n]) / np.sqrt((i - n[0]) ** 2 + (j - n[1]) ** 2)
                    valid_neighbors.append((n, slope))

            total_slope = sum(s for _, s in valid_neighbors)
            if total_slope > 0:
                for n, slope in valid_neighbors:
                    D[n] += (E[i, j] * slope) / total_slope  # 按坡度比例分配沉积

    return D

D = distribute_sediment(E, DEM, DAMS)  # 计算沉积量

# === 计算植被 SOC 输入 ===
def vegetation_input(NDVI, base_input=0.01, max_input=0.1):
    NDVI = NDVI / 10000.0  # Normalize NDVI
    return max_input * (1 - np.exp(-NDVI / base_input))

V = vegetation_input(NDVI)  # 计算植被输入量


# === 3. SOC 反应-迁移模型（包含快分解和慢分解 SOC）===
def soc_dynamic_model(C_fast, C_slow, E, D, V, T, SM, NDVI, k_base, dx, dt, p_fast=0.3):
    """
    SOC 反应-迁移模型，包含快分解和慢分解部分
    - C_fast: 快分解 SOC
    - C_slow: 慢分解 SOC
    - p_fast: 快分解 SOC 占比
    """

    # 计算快分解和慢分解的分解速率
    k_fast = k_base * 5 * (1 + 0.01 * T) * (1 + 0.05 * SM) * (1 - NDVI)
    k_slow = k_base * 0.1 * (1 + 0.01 * T) * (1 + 0.05 * SM) * (1 - NDVI)

    # 侵蚀项
    erosion_fast = -E * C_fast
    erosion_slow = -E * C_slow

    # 沉积项
    deposition_fast = D * p_fast
    deposition_slow = D * (1 - p_fast)

    # 植被输入
    vegetation_fast = V * p_fast
    vegetation_slow = V * (1 - p_fast)

    # 矿化分解项
    reaction_fast = -k_fast * C_fast
    reaction_slow = -k_slow * C_slow

    # 计算新 SOC
    C_fast_new = C_fast + (erosion_fast + deposition_fast + vegetation_fast + reaction_fast) * dt
    C_slow_new = C_slow + (erosion_slow + deposition_slow + vegetation_slow + reaction_slow) * dt

    # 确保 SOC 不为负
    C_fast_new = np.maximum(C_fast_new, 0)
    C_slow_new = np.maximum(C_slow_new, 0)

    return C_fast_new, C_slow_new


# === 4. 运行 100 年 SOC 变化模拟 ===
k_base = 0.001
dx = 10
dt = 1
timesteps = 100

T_series = [create_grid(df, temp_col) + np.random.randn(*grid_size) * 1.5 for _ in range(timesteps)]
SM_series = [create_grid(df, moisture_col) + np.random.randn(*grid_size) * 5 for _ in range(timesteps)]

p_fast = 0.3  # 快分解 SOC 占比
C_fast = C * p_fast  # 快分解 SOC
C_slow = C * (1 - p_fast)  # 慢分解 SOC

for t in range(timesteps):
    if t > 0:
        T = T_series[t]
        SM = SM_series[t]
        C_fast, C_slow = soc_dynamic_model(C_fast, C_slow, E, D, V, T, SM, NDVI, k_base, dx, dt, p_fast)

    if t <= 10:  # 仅绘制前 10 年
        plt.imshow(C_fast + C_slow, cmap="viridis", extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        plt.colorbar(label="SOC Concentration (g/kg)")
        plt.title(f"SOC Concentration at Year {t}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()


