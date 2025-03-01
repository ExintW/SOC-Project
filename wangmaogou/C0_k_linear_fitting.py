import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

# ✅ 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 📥 读取 CSV 文件
# -----------------------------
# ✅ 替换为你的 CSV 文件路径
file_path = "D://EcoSci//Dr.Shi//Data//fit_results.csv"  # 📂 替换为实际文件路径
df = pd.read_csv(file_path)

# 🚀 数据预处理
df['Rainfall Intensity'] = df['Rainfall Intensity'].str.replace('mm/h', '').astype(float)  # 转换为数值

# 📊 按区域分组
erosion_df = df[df['Area'] == 'Erosion Area']
sedimentation_df = df[df['Area'] == 'Sedimentation Area']

# -----------------------------
# 📈 拟合函数定义
# -----------------------------
def linear_model(x, a, b):
    return a * x + b

def perform_fit_and_plot(area_name, data, axes_row):
    rainfall = data['Rainfall Intensity'].values
    c0_values = data['c0'].values
    k_values = data['k'].values

    # ✅ c0 和 k 的线性拟合
    popt_c0, _ = curve_fit(linear_model, rainfall, c0_values)
    popt_k, _ = curve_fit(linear_model, rainfall, k_values)

    # ✅ 样条插值
    c0_spline = CubicSpline(rainfall, c0_values)
    k_spline = CubicSpline(rainfall, k_values)

    # 📈 绘制拟合曲线
    rainfall_fit = np.linspace(min(rainfall) - 10, max(rainfall) + 10, 200)
    c0_fit_linear = linear_model(rainfall_fit, *popt_c0)
    k_fit_linear = linear_model(rainfall_fit, *popt_k)
    c0_fit_spline = c0_spline(rainfall_fit)
    k_fit_spline = k_spline(rainfall_fit)

    # -------- c0 拟合图 --------
    axes_row[0].scatter(rainfall, c0_values, color='black', label='原始数据', zorder=5)
    axes_row[0].plot(rainfall_fit, c0_fit_linear, 'r--', label='线性拟合')
    axes_row[0].plot(rainfall_fit, c0_fit_spline, 'g-', label='样条插值')
    axes_row[0].set_title(f'{area_name} - 降雨强度与 $c_0$ 关系')
    axes_row[0].set_xlabel('降雨强度 (mm/h)')
    axes_row[0].set_ylabel('$c_0$')
    axes_row[0].legend()
    axes_row[0].grid(True)

    # -------- k 拟合图 --------
    axes_row[1].scatter(rainfall, k_values, color='black', label='原始数据', zorder=5)
    axes_row[1].plot(rainfall_fit, k_fit_linear, 'r--', label='线性拟合')
    axes_row[1].plot(rainfall_fit, k_fit_spline, 'g-', label='样条插值')
    axes_row[1].set_title(f'{area_name} - 降雨强度与 $k$ 关系')
    axes_row[1].set_xlabel('降雨强度 (mm/h)')
    axes_row[1].set_ylabel('$k$')
    axes_row[1].legend()
    axes_row[1].grid(True)

    # ✅ 打印拟合公式
    print(f"✅ {area_name} - c0 线性拟合公式: c0 = {popt_c0[0]:.8f} * rainfall + {popt_c0[1]:.8f}")
    print(f"✅ {area_name} - k 线性拟合公式: k = {popt_k[0]:.8f} * rainfall + {popt_k[1]:.8f}\n")

# -----------------------------
# 📈 绘图与拟合执行
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # 2 行 2 列子图

# 侵蚀区拟合
perform_fit_and_plot("侵蚀区", erosion_df, axes[0])

# 沉积区拟合
perform_fit_and_plot("沉积区", sedimentation_df, axes[1])

plt.tight_layout()
plt.show()
