import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置支持中文字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 处理负号显示问题

# 读取CSV文件
file_path = r"D:\EcoSci\Dr.Shi\Data\Water_sediment_process.csv"
df = pd.read_csv(file_path)

# 提取并清理降雨强度列 (去掉 'mm/h' 并转为 float)
df['Rainfall Intensity (mm/h)'] = df.iloc[:, 0].str.replace('mm/h', '').astype(float)

# 提取需要的列
rainfall_intensity = df['Rainfall Intensity (mm/h)']
flow_rate = df['Flow Rate (L/min)']

# 线性拟合
linear_coeffs = np.polyfit(rainfall_intensity, flow_rate, 1)
linear_fit = np.poly1d(linear_coeffs)
linear_eq = f"线性拟合: y = {linear_coeffs[0]:.6f}x + {linear_coeffs[1]:.6f}"

# 二次多项式拟合 (使用 ^2 替代平方符号)
poly_coeffs = np.polyfit(rainfall_intensity, flow_rate, 2)
poly_fit = np.poly1d(poly_coeffs)
poly_eq = f"二次拟合: y = {poly_coeffs[0]:.6f}x^2 + {poly_coeffs[1]:.6f}x + {poly_coeffs[2]:.6f}"

# 控制台输出拟合公式
print("拟合公式：")
print(linear_eq)
print(poly_eq)

# 生成拟合曲线 x 值
x_fit = np.linspace(rainfall_intensity.min(), rainfall_intensity.max(), 100)
linear_y_fit = linear_fit(x_fit)
poly_y_fit = poly_fit(x_fit)

# 绘制图表
plt.figure(figsize=(10, 6))
plt.scatter(rainfall_intensity, flow_rate, color='blue', s=100, label='数据点')

# 绘制拟合曲线
plt.plot(x_fit, linear_y_fit, '--', label=linear_eq, linewidth=2)
plt.plot(x_fit, poly_y_fit, '-', color='red', label=poly_eq, linewidth=2)

# 标注公式在图中
plt.text(0.05, 0.95, linear_eq, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', color='blue')
plt.text(0.05, 0.88, poly_eq, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', color='red')

# 图表设置
plt.xlabel('降雨强度 (mm/h)', fontsize=12)
plt.ylabel('流量 (L/min)', fontsize=12)
plt.title('降雨强度与流量的关系', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()




