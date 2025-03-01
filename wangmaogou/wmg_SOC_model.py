import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# === 1️⃣ 数据读取 ===
region_csv_path = r'D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_wangmaogou_1km_with_DEM_region_labeled.csv'
dam_excel_path = r'D:\EcoSci\Dr.Shi\Data\wangmaogou_dams.xlsx'
proportion_csv_path = r'D:\EcoSci\Dr.Shi\Data\Fast_Slow_SOC_Proportion.csv'

df = pd.read_csv(region_csv_path, encoding='utf-8-sig')
df_dam = pd.read_excel(dam_excel_path)
df_prop = pd.read_csv(proportion_csv_path, encoding='utf-8-sig')

# === 2️⃣ 网格生成与变量提取 ===
lon_col, lat_col = "LON", "LAT"
soc_col, temp_col, moisture_col, ndvi_col, dem_col, rain_col, landuse_col, region_col, slope_col = (
    "ORGA", "TEMP", "MOISTURE", "NDVI", "htgy_DEM", "RAIN", "LANDUSE", "Region", "SLOPE"
)

grid_x = np.sort(df[lon_col].unique())
grid_y = np.sort(df[lat_col].unique())[::-1]
grid_size = (len(grid_y), len(grid_x))

def create_grid(data, col_name):
    return data.pivot(index=lat_col, columns=lon_col, values=col_name).sort_index(ascending=False).values

C = create_grid(df, soc_col)
DEM = create_grid(df, dem_col)
NDVI = create_grid(df, ndvi_col)
RAIN = create_grid(df, rain_col)
SAND = create_grid(df, "SAND")
SILT = create_grid(df, "SILT")
CLAY = create_grid(df, "CLAY")
LANDUSE = create_grid(df, landuse_col)
REGION = create_grid(df, region_col)
SLOPE = create_grid(df, slope_col)

# === 3️⃣ 快慢 SOC 分配 (基于土地利用) ===
def allocate_fast_slow_soc(C, LANDUSE, proportion_df):
    prop_dict = {row['Type']: {'fast': row['Fast SOC(%)'] / 100, 'slow': row['Slow SOC(%)'] / 100}
                 for _, row in proportion_df.iterrows()}

    rows, cols = LANDUSE.shape
    C_fast, C_slow = np.zeros((rows, cols)), np.zeros((rows, cols))
    p_fast_grid = np.zeros((rows, cols))  # NEW: Fast SOC proportion grid

    for i in range(rows):
        for j in range(cols):
            land_type = LANDUSE[i, j]
            props = prop_dict.get(land_type, {'fast': 0, 'slow': 1})
            C_fast[i, j] = C[i, j] * props['fast']
            C_slow[i, j] = C[i, j] * props['slow']
            p_fast_grid[i, j] = props['fast']  # Save fast SOC proportion

    return C_fast, C_slow, p_fast_grid


C_fast, C_slow, p_fast_grid = allocate_fast_slow_soc(C, LANDUSE, df_prop)


# === 4️⃣ 大坝数据处理 ===
df_dam["Total Capacity (L)"] = df_dam["Total Capacity (10,000 m³)"] * 10_000 * 1000
df_dam["Current Capacity (L)"] = df_dam["Current Capacity (10,000 m³)"] * 10_000 * 1000
df_dam["Capacity Remained (L)"] = df_dam["Total Capacity (L)"] - df_dam["Current Capacity (L)"]

def find_nearest_index(array, value):
    return (np.abs(array - value)).argmin()

dam_positions = [
    {'i': find_nearest_index(grid_y, row["Latitude"]), 'j': find_nearest_index(grid_x, row["Longitude"]),
     'capacity_remained': row["Capacity Remained (L)"]}
    for _, row in df_dam.iterrows()
]

# === 5️⃣ RUSLE 土壤流失量 (E) 计算 ===
def calculate_r_factor(rainfall):
    return 1.735 * rainfall * 365

def calculate_ls_factor(slope, slope_length=1000):
    slope_rad = np.deg2rad(slope)
    return ((slope_length / 22.13) ** 0.4) * ((np.sin(slope_rad) / 0.0896) ** 1.3)

def calculate_k_factor(sand, silt):
    """土壤可蚀性因子 K (t·ha·h/(MJ·mm·ha))"""
    return 0.0034 + 0.0405 * np.exp(-0.5 * (((silt + sand)/100 - 0.5)**2))


def calculate_c_factor(ndvi):
    return np.exp(-2.5 * ndvi)

def calculate_p_factor(landuse):
    p_values = {"Sloping cropland": 0.6, "Forestland": 0.4, "Grassland": 0.5, "Terrace": 0.1, "Dam field": 0.05}
    return p_values.get(landuse, 1.0)

# 因子计算
R = calculate_r_factor(RAIN)
K = np.full_like(R, 0.03)
LS = calculate_ls_factor(SLOPE)
C_factor = calculate_c_factor(NDVI)
P = np.array([[calculate_p_factor(LANDUSE[i, j]) for j in range(LANDUSE.shape[1])] for i in range(LANDUSE.shape[0])])

E = (R * K * LS * C_factor * P) / 365  # t/ha/day

# === 6️⃣ 土壤流失量转 SOC 流失量 (g/kg/day) ===
def convert_soil_loss_to_soc_loss(E, ORGA, bulk_density=1300):
    E_g_m2_day = E * 100
    soc_loss_g_m2_day = E_g_m2_day * (ORGA / 1000) * bulk_density
    soc_loss_g_kg_day = soc_loss_g_m2_day / bulk_density
    return soc_loss_g_kg_day

soc_loss = convert_soil_loss_to_soc_loss(E, C)

# === 7️⃣ 沉积分配 (考虑大坝库容和流量) ===
def calculate_flowrate(rain_intensity):
    return 0.027884 * rain_intensity + 0.014370

def distribute_sediment_with_dams(E, DEM, dam_positions, RAIN, dt=1):
    D = np.zeros_like(E, dtype=float)
    rows, cols = DEM.shape
    dam_capacity_map = {(dam['i'], dam['j']): dam['capacity_remained'] for dam in dam_positions}

    for i in range(rows):
        for j in range(cols):
            soc_flow = E[i, j]
            if soc_flow <= 0:
                continue

            if (i, j) in dam_capacity_map and dam_capacity_map[(i, j)] > 0:
                D[i, j] += soc_flow
                flowrate = calculate_flowrate(RAIN[i, j]) * 60 * 24 * dt
                dam_capacity_map[(i, j)] = max(dam_capacity_map[(i, j)] - flowrate, 0)
                continue

            neighbors = [(i + di, j + dj) for di in [-1, 0, 1] for dj in [-1, 0, 1] if (di, dj) != (0, 0)]
            valid_neighbors = [(n, (DEM[i, j] - DEM[n[0], n[1]]) / np.hypot(i - n[0], j - n[1]))
                               for n in neighbors if 0 <= n[0] < rows and 0 <= n[1] < cols and DEM[n[0], n[1]] < DEM[i, j]]
            total_slope = sum(s for _, s in valid_neighbors)
            if total_slope > 0:
                for (ni, nj), slope in valid_neighbors:
                    D[ni, nj] += (soc_flow * slope) / total_slope
    return D

D = distribute_sediment_with_dams(soc_loss, DEM, dam_positions, RAIN)

# === 8️⃣ 植被输入 ===
def vegetation_input(NDVI):
    return 0.00000121 * np.exp(0.08077438 * NDVI) - 0.00012108

V = vegetation_input(NDVI)

# === 9️⃣ Reaction 计算 (基于区域与降雨) ===
def calculate_c0_k(rainfall, region_type):
    """计算 c0 和 k 参数"""
    if region_type == "erosion area":
        return 0.01588787 * rainfall + 0.91032533, 0.00026035 * rainfall + 0.03905750
    elif region_type == "sedimentation area":
        return 0.00660670 * rainfall + 1.17265400, 0.00004263 * rainfall + 0.05567833
    return 0, 0

def generate_new_reaction(C, RAIN, REGION):
    """生成新的降雨反应 (c0 和 k 分布)"""
    rows, cols = C.shape
    c0_map = np.zeros_like(C)
    k_map = np.zeros_like(C)

    for i in range(rows):
        for j in range(cols):
            c0, k = calculate_c0_k(RAIN[i, j], REGION[i, j])
            c0_map[i, j] = c0
            k_map[i, j] = k

    return {"days_passed": 0, "reaction_map": c0_map, "decay_rate": k_map}

def update_reactions(reaction_effects):
    """更新所有反应并计算总反应 (dc/dt)"""
    total_reaction = np.zeros_like(reaction_effects[0]["reaction_map"]) if reaction_effects else 0

    # 更新所有反应
    updated_effects = []
    for effect in reaction_effects:
        days = effect["days_passed"]
        c0_map = effect["reaction_map"]
        k_map = effect["decay_rate"]

        # 计算当天分解贡献
        daily_reaction = -c0_map * np.exp(-k_map * days)
        total_reaction += daily_reaction

        # 如果未超过 56 天，保留反应
        if days < 56:
            updated_effects.append({"days_passed": days + 1, "reaction_map": c0_map, "decay_rate": k_map})

    return total_reaction, updated_effects


def reaction_model(C, RAIN, REGION, t):
    rows, cols = C.shape
    reaction = np.zeros_like(C)
    for i in range(rows):
        for j in range(cols):
            c0, k = calculate_c0_k(RAIN[i, j], REGION[i, j])
            reaction[i, j] = -c0 * np.exp(-k * t)
    return reaction

# === 🔟 SOC 动态模型 ===
def soc_dynamic_model(C_fast, C_slow, soc_loss_g_kg_day, D, V, RAIN, REGION, p_fast_grid, dt, t,
                      reaction_effects_fast=[], reaction_effects_slow=[]):
    # ✅ Erosion
    erosion_fast = -soc_loss_g_kg_day * p_fast_grid
    erosion_slow = -soc_loss_g_kg_day * (1 - p_fast_grid)

    # ✅ Deposition
    deposition_fast = D * p_fast_grid
    deposition_slow = D * (1 - p_fast_grid)

    # ✅ Vegetation input
    vegetation_fast = V * p_fast_grid
    vegetation_slow = V * (1 - p_fast_grid)

    # ✅ Generate new reactions
    new_reaction_fast = generate_new_reaction(C_fast, RAIN, REGION)
    new_reaction_slow = generate_new_reaction(C_slow, RAIN, REGION)
    reaction_effects_fast.append(new_reaction_fast)
    reaction_effects_slow.append(new_reaction_slow)

    # ✅ Update reactions
    reaction_fast, reaction_effects_fast = update_reactions(reaction_effects_fast)
    reaction_slow, reaction_effects_slow = update_reactions(reaction_effects_slow)

    # ✅ Update SOC concentrations
    C_fast_new = np.maximum(C_fast + (erosion_fast + deposition_fast + vegetation_fast + reaction_fast) * dt, 0)
    C_slow_new = np.maximum(C_slow + (erosion_slow + deposition_slow + vegetation_slow + reaction_slow) * dt, 0)

    return C_fast_new, C_slow_new, reaction_effects_fast, reaction_effects_slow


# 🕹️ 初始化反应队列
reaction_effects_fast = []
reaction_effects_slow = []

dt, timesteps = 1, 100

for t in range(timesteps):
    if t > 0:
        C_fast, C_slow, reaction_effects_fast, reaction_effects_slow = soc_dynamic_model(
            C_fast, C_slow, soc_loss, D, V, RAIN, REGION, p_fast_grid, dt, t,
            reaction_effects_fast=reaction_effects_fast,
            reaction_effects_slow=reaction_effects_slow
        )
    if t <= 10:
        fig, ax = plt.subplots()

        cax = ax.imshow(C_fast + C_slow, cmap="viridis",
                        extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        cbar = fig.colorbar(cax, label="SOC Concentration (g/kg)")

        ax.set_title(f"SOC Concentration at Year {t}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # ✅ 设置经度刻度格式为非科学计数法
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style='plain', axis='x')  # 另一种方式确保非科学计数法

        plt.show()

