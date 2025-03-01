import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
from matplotlib import font_manager

# -----------------------------
# 🖥️ 设置 matplotlib 支持中文和负号
# -----------------------------
font_candidates = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'STHeiti', 'PingFang SC']
font_found = False

for font in font_candidates:
    if font in [f.name for f in font_manager.fontManager.ttflist]:
        matplotlib.rcParams['font.family'] = font
        font_found = True
        break

if not font_found:
    print("⚠️ 未找到常用中文字体，请安装 'Microsoft YaHei' 或 'SimHei' 以显示中文。")

matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# -----------------------------
# 📥 读取 CSV 文件
# -----------------------------
file_path = r"D:\EcoSci\Dr.Shi\Data\SOC_Mineralization_Data.csv"  # 修改为你的实际路径
df = pd.read_csv(file_path)

# 📂 拟合参数保存列表
fit_results = []

# -----------------------------
# 📊 数据提取函数 (已修复排序问题)
# -----------------------------
def extract_data(df, area, rainfall_intensity):
    intensity_rows = {
        "60mm/h": (1, 6),   # CSV 第2到6行 (Python 索引从0开始)
        "90mm/h": (6, 11),  # 第7到11行
        "120mm/h": (11, 16) # 第12到16行
    }

    if rainfall_intensity not in intensity_rows:
        raise ValueError(f"❌ 无效降雨强度: {rainfall_intensity}")

    start_row, end_row = intensity_rows[rainfall_intensity]

    # 提取天数和矿化量
    days = pd.to_numeric(df.iloc[start_row:end_row, 1], errors='coerce').values
    if area == "Erosion Area":
        mineralization = pd.to_numeric(df.iloc[start_row:end_row, 4], errors='coerce').values
    elif area == "Sedimentation Area":
        mineralization = pd.to_numeric(df.iloc[start_row:end_row, 6], errors='coerce').values
    else:
        raise ValueError("❌ 区域无效，请输入 'Erosion Area' 或 'Sedimentation Area'。")

    # ✅ 按天数升序排列
    sort_index = np.argsort(days)
    days_sorted = days[sort_index]
    mineralization_sorted = mineralization[sort_index]

    # 🖨️ 打印提取的数据
    print(f"\n📊 提取数据 - {area} | {rainfall_intensity}:")
    for day, value in zip(days_sorted, mineralization_sorted):
        print(f"  - Day {int(day)}: {value:.6f}")

    return days_sorted, mineralization_sorted

# -----------------------------
# 🧮 指数衰减模型与拟合函数
# -----------------------------
def exp_decay_model(t, c0, k):
    return c0 * np.exp(-k * t)

def fit_exp_decay(t, dc_dt):
    if len(t) != len(dc_dt):
        raise ValueError("❌ 天数和矿化量数据长度不匹配。")
    popt, _ = curve_fit(exp_decay_model, t, dc_dt, p0=(2.0, 0.05), maxfev=10000)
    return popt

# -----------------------------
# 📈 拟合、打印结果、绘图并保存结果
# -----------------------------
def process_and_plot(area):
    colors = {"60mm/h": "blue", "90mm/h": "green", "120mm/h": "red"}
    markers = {"60mm/h": "o", "90mm/h": "s", "120mm/h": "D"}
    plt.figure(figsize=(12, 8))
    t_fit = np.linspace(0, 60, 200)  # 平滑曲线时间范围

    print(f"\n🔍 {area} 拟合与数据提取结果:")

    for intensity in ["60mm/h", "90mm/h", "120mm/h"]:
        try:
            # 提取数据
            t_values, dc_dt = extract_data(df, area, intensity)

            # 拟合模型
            c0, k = fit_exp_decay(t_values, dc_dt)
            dc_dt_fit = exp_decay_model(t_fit, c0, k)

            # ✅ 保存拟合参数到列表
            fit_results.append({
                "Area": area,
                "Rainfall Intensity": intensity,
                "c0": round(c0, 6),
                "k": round(k, 6)
            })

            # 🖨️ 打印拟合公式
            print(f"✅ {intensity} 拟合公式: dc/dt = {c0:.6f} * exp(-{k:.6f} * t)\n")

            # 📊 绘制原始数据与拟合曲线
            plt.scatter(t_values, dc_dt, color=colors[intensity], marker=markers[intensity],
                        label=f"{intensity} 数据", s=90, edgecolors='k', linewidth=1.2)
            plt.plot(t_fit, dc_dt_fit, color=colors[intensity], linestyle='--',
                     label=f"{intensity} 拟合: c0={c0:.4f}, k={k:.4f}", linewidth=2)

        except Exception as e:
            print(f"❌ {intensity} 拟合失败: {e}")

    # 📈 图表设置
    plt.xlabel('培养天数 (t)', fontsize=14)
    plt.ylabel('矿化量 (dc/dt)', fontsize=14)
    plt.title(f'{area} SOC 矿化量指数衰减拟合', fontsize=16)
    plt.legend(fontsize=10, loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 🚀 执行拟合并绘图
# -----------------------------
process_and_plot("Erosion Area")        # 侵蚀区拟合与绘图
process_and_plot("Sedimentation Area")  # 沉积区拟合与绘图

# -----------------------------
# 💾 保存拟合结果到 CSV 文件
# -----------------------------
output_df = pd.DataFrame(fit_results)
output_path = r"D:\EcoSci\Dr.Shi\Data\fit_results.csv"  # 修改为你的保存路径
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 拟合参数已保存至: {output_path}")
