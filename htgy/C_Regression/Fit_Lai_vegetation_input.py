import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import re
import sys
import os

# 若有上级目录模块依赖，可自行修改此路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# --------------------------------------------------
#                数据读取与清洗
# --------------------------------------------------

# 示例中请替换为你真实的 CSV 文件路径
# 如果已有 DATA_DIR / "Vegetation_Input_v2_with_Lai.csv"，请保留
from htgy.globals import *
csv_file = DATA_DIR / "Vegetation_Input_v2_with_Lai.csv"
df = pd.read_csv(csv_file)

# 标准化列名（去掉首尾空格）
df.columns = df.columns.str.strip()

# 定义相关列名称
lai_col = "Lai"
soc_col = "Monthly Increase in Soil Organic Carbon (g/kg)"

# 检查原始 SOC 列前 10 行（可能带有特殊字符）
print("Original SOC values before cleaning:\n", df[soc_col].head(10))


# 清洗函数：移除字符串中的特殊字符，只保留数字、点和减号
def clean_numeric(value):
    if isinstance(value, str):
        value = re.sub(r'[^\d\.-]', '', value)  # 只保留数字、点和减号
        try:
            return float(value) if value else np.nan
        except ValueError:
            return np.nan
    return value


# 对 LAI 和 SOC 列应用清洗函数
df[soc_col] = df[soc_col].apply(clean_numeric)
df[lai_col] = df[lai_col].apply(clean_numeric)

# 检查清洗后的 SOC 列前 10 行
print("Cleaned SOC values:\n", df[soc_col].head(10))

# 删除 NaN 数据（但保留 0 和负值）
df_cleaned = df.dropna(subset=[lai_col, soc_col])

# 分析正值和负值的点数
negative_points = df_cleaned[df_cleaned[soc_col] < 0]
positive_points = df_cleaned[df_cleaned[soc_col] > 0]
print(f"Total data points: {len(df_cleaned)}")
print(f"Positive SOC points: {len(positive_points)}")
print(f"Zero or negative SOC points: {len(negative_points)}")

# 提取用于拟合的 X (LAI) 和 Y (SOC)
X = df_cleaned[lai_col].values
Y = df_cleaned[soc_col].values


# --------------------------------------------------
#              定义各种回归模型
# --------------------------------------------------

# 1. 线性模型: y = a * x + b
def linear_model(x, a, b):
    return a * x + b


# 2. 指数模型: y = a * exp(b * x)
def exponential_model(x, a, b):
    return a * np.exp(b * x)


# 3. 幂函数模型: y = a * x^b
def power_model(x, a, b):
    return a * np.power(x, b)


# 4. 对数模型: y = a * log(x) + b
def logarithmic_model(x, a, b):
    return a * np.log(x) + b


# 5. 二次多项式模型: y = a + b*x + c*x^2
def polynomial2_model(x, a, b, c):
    return a + b * x + c * (x ** 2)


# --------------------------------------------------
#            误差分析：MSE, RMSE, R^2
# --------------------------------------------------
def compute_errors(x, y, model_func, params):
    """
    传入:
        x: X数据
        y: Y数据
        model_func: 拟合函数
        params: 拟合得到的参数列表
    返回:
        mse: 均方误差
        rmse: 均方根误差
        r2: 决定系数 R^2
    """
    y_pred = model_func(x, *params)
    residuals = y - y_pred
    sse = np.sum(residuals ** 2)  # SSE: Sum of Squared Errors
    mse = sse / len(x)  # MSE: Mean Squared Error
    rmse = np.sqrt(mse)  # RMSE
    tss = np.sum((y - np.mean(y)) ** 2)  # TSS: Total Sum of Squares
    r2 = 1 - sse / tss if tss != 0 else 0  # R^2
    return mse, rmse, r2


# --------------------------------------------------
#             拟合并打印方程/误差结果
# --------------------------------------------------

models = {
    "Linear": linear_model,
    "Exponential": exponential_model,
    "Power-Law": power_model,
    "Logarithmic": logarithmic_model,
    "Polynomial_2": polynomial2_model
}

fit_results = {}  # 存储各模型的拟合参数
error_results = {}  # 存储各模型的误差结果

for name, model_func in models.items():
    try:
        # 使用 curve_fit 进行拟合
        # 注意：某些模型（如对数模型）需要 x>0，否则可能出错
        params, _ = curve_fit(model_func, X, Y, maxfev=5000)
        fit_results[name] = params

        # 计算误差指标
        mse, rmse, r2 = compute_errors(X, Y, model_func, params)
        error_results[name] = (mse, rmse, r2)

        # 打印方程和误差信息
        if name == "Linear":
            eq_str = f"y = {params[0]:.8f}*x + {params[1]:.8f}"
        elif name == "Exponential":
            eq_str = f"y = {params[0]:.8f} * exp({params[1]:.8f}*x)"
        elif name == "Power-Law":
            eq_str = f"y = {params[0]:.8f} * x^{params[1]:.8f}"
        elif name == "Logarithmic":
            eq_str = f"y = {params[0]:.8f} * ln(x) + {params[1]:.8f}"
        elif name == "Polynomial_2":
            eq_str = f"y = {params[0]:.8f} + {params[1]:.8f}*x + {params[2]:.8f}*x^2"
        else:
            eq_str = "方程格式未定义"

        print(f"{name} Model:")
        print(f"  Equation: {eq_str}")
        print(f"  MSE  = {mse:.8f}")
        print(f"  RMSE = {rmse:.8f}")
        print(f"  R^2  = {r2:.8f}")
        print("-" * 50)

    except Exception as e:
        print(f"Could not fit {name} model: {e}")
        print("-" * 50)

# --------------------------------------------------
#                 绘图与可视化
# --------------------------------------------------

# 为绘图生成等间距的 x 值
x_vals = np.linspace(X.min(), X.max(), 100)

plt.figure(figsize=(10, 6))
# 分别画正值和负值的散点
sns.scatterplot(x=positive_points[lai_col], y=positive_points[soc_col],
                color="blue", alpha=0.7, label="Positive SOC")
sns.scatterplot(x=negative_points[lai_col], y=negative_points[soc_col],
                color="red", alpha=0.7, label="Negative SOC")

# 将每个模型的拟合曲线画出来并在图中标注方程
for name, model_func in models.items():
    if name in fit_results:
        params = fit_results[name]
        y_fit = model_func(x_vals, *params)
        plt.plot(x_vals, y_fit, label=f"{name} Fit")

        # 方程简写（为了在图上标注，减少长度）
        if name == "Linear":
            eq_text = f"y={params[0]:.4f}x+{params[1]:.4f}"
        elif name == "Exponential":
            eq_text = f"y={params[0]:.4f}e^({params[1]:.4f}x)"
        elif name == "Power-Law":
            eq_text = f"y={params[0]:.4f}x^{params[1]:.4f}"
        elif name == "Logarithmic":
            eq_text = f"y={params[0]:.4f}ln(x)+{params[1]:.4f}"
        elif name == "Polynomial_2":
            eq_text = f"y={params[0]:.4f}+{params[1]:.4f}x+{params[2]:.4f}x^2"
        else:
            eq_text = "Unknown"

        # 在曲线靠右侧(80%处)位置标注文字
        x_pos = np.percentile(x_vals, 80)
        y_pos = model_func(x_pos, *params) * 1.1  # 略微上移，防止文字和曲线重叠
        plt.text(x_pos, y_pos, eq_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel("Leaf Area Index (LAI)")
plt.ylabel("Monthly Increase in Soil Organic Carbon (g/kg)")
plt.title("Empirical Fit of LAI vs. SOC Increase (Including Negative Values)")
plt.legend()
plt.grid(True)
plt.show()
