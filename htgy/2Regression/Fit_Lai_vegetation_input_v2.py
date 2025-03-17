import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import re
import os
import sys
from htgy.globals import *  # 假设该模块定义了 PROCESSED_DIR 等全局变量

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -----------------------------------------------------------
# 1. 读取包含5个sheet的Excel文件，并合并成一个DataFrame
# -----------------------------------------------------------
excel_file = PROCESSED_DIR / "River_Basin_Points_Updated.xlsx"
# 使用pandas读取Excel文件
xls = pd.ExcelFile(excel_file)
all_sheets = xls.sheet_names

# 遍历所有sheet，将每个sheet的数据读取并合并
df_list = []
for sheet in all_sheets:
    df_sheet = pd.read_excel(xls, sheet_name=sheet)
    df_sheet["SheetName"] = sheet  # 可选：记录数据来源sheet名称
    df_list.append(df_sheet)

df = pd.concat(df_list, ignore_index=True)
# 去除列名中的多余空格
df.columns = df.columns.str.strip()

# -----------------------------------------------------------
# 2. 数据预处理：选择相关列并清洗数值数据
# -----------------------------------------------------------
# 定义独立变量和因变量
lai_col = "2007 LAI"  # 独立变量
soc_col = "SOC Monthly Increase (g/kg/month)"  # 因变量

print("预览选定的列数据：")
print(df[[lai_col, soc_col]].head())


def clean_numeric(value):
    """
    清洗数值数据：
    - 对字符串，去除非数字、小数点和负号的字符
    - 转换为float，若转换失败则返回NaN
    """
    if isinstance(value, str):
        # 使用正则表达式去除无关字符
        value = re.sub(r'[^\d\.-]', '', value)
        try:
            return float(value) if value else np.nan
        except ValueError:
            return np.nan
    return value


# 对指定列应用数据清洗函数
df[soc_col] = df[soc_col].apply(clean_numeric)
df[lai_col] = df[lai_col].apply(clean_numeric)

# 删除独立变量或因变量为空的数据
df_cleaned = df.dropna(subset=[lai_col, soc_col])

# -----------------------------------------------------------
# 2.1 可选：过滤掉SOC值大于指定阈值的数据
# -----------------------------------------------------------
apply_threshold_filter = True  # 设置为True则应用过滤
threshold_value = 0.15  # 阈值
if apply_threshold_filter:
    df_cleaned = df_cleaned[df_cleaned[soc_col] <= threshold_value]

# -----------------------------------------------------------
# 3. 数据分离：将正值和负值数据点分别存储，便于后续绘图
# -----------------------------------------------------------
negative_points = df_cleaned[df_cleaned[soc_col] < 0]
positive_points = df_cleaned[df_cleaned[soc_col] > 0]
print(f"数据总数: {len(df_cleaned)}")
print(f"SOC正增量数据点: {len(positive_points)}")
print(f"SOC非正（零或负）数据点: {len(negative_points)}")

# 提取用于曲线拟合的数组
X = df_cleaned[lai_col].values
Y = df_cleaned[soc_col].values


# -----------------------------------------------------------
# 4. 定义经验模型函数
# -----------------------------------------------------------
def linear_model(x, a, b):
    """线性模型: y = a * x + b"""
    return a * x + b


def exponential_model(x, a, b):
    """指数模型: y = a * exp(b * x)"""
    return a * np.exp(b * x)


def power_model(x, a, b):
    """幂函数模型: y = a * x^b"""
    return a * np.power(x, b)


def logarithmic_model(x, a, b):
    """对数模型: y = a * log(x) + b"""
    return a * np.log(x) + b


# 将模型存入字典，便于遍历
models = {
    "Linear": linear_model,
    "Exponential": exponential_model,
    "Power-Law": power_model,
    "Logarithmic": logarithmic_model
}

fit_results = {}  # 用于保存每个模型的拟合参数

# -----------------------------------------------------------
# 5. 使用curve_fit对每个模型进行拟合
# -----------------------------------------------------------
print("\n模型拟合结果：")
for name, model in models.items():
    try:
        # 进行曲线拟合，设置maxfev以增加迭代次数防止不收敛
        params, _ = curve_fit(model, X, Y, maxfev=5000)
        fit_results[name] = params
        # 格式化并打印拟合的方程
        if name == "Linear":
            equation_text = f"y = {params[0]:.8f} * LAI + {params[1]:.8f}"
        elif name == "Exponential":
            equation_text = f"y = {params[0]:.8f} * exp({params[1]:.8f} * LAI)"
        elif name == "Power-Law":
            equation_text = f"y = {params[0]:.8f} * LAI^{params[1]:.8f}"
        elif name == "Logarithmic":
            equation_text = f"y = {params[0]:.8f} * log(LAI) + {params[1]:.8f}"
        print(f"{name}模型: {equation_text}")
    except Exception as e:
        print(f"无法拟合 {name} 模型: {e}")

# -----------------------------------------------------------
# 5.1 误差分析：计算每个模型的残差平方和、均方误差、RMSE及R²
# -----------------------------------------------------------
print("\n模型误差分析：")
error_analysis = {}
for name, model in models.items():
    if name in fit_results:
        params = fit_results[name]
        # 使用拟合参数计算预测值
        y_pred = model(X, *params)
        # 计算残差
        residuals = Y - y_pred
        # 残差平方和 (Sum of Squared Residuals)
        ss_res = np.sum(residuals ** 2)
        # 总平方和 (Total Sum of Squares)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        # R²计算，防止除0错误
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
        # 均方误差（MSE）和均方根误差（RMSE）
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        error_analysis[name] = {"SS_res": ss_res, "MSE": mse, "RMSE": rmse, "R2": r_squared}
        print(f"{name}模型: SS_res = {ss_res:.6f}, MSE = {mse:.6f}, RMSE = {rmse:.6f}, R² = {r_squared:.6f}")

# -----------------------------------------------------------
# 6. 绘制数据点及拟合曲线图
# -----------------------------------------------------------
# 生成用于绘图的x值序列
x_vals = np.linspace(X.min(), X.max(), 100)

plt.figure(figsize=(10, 6))

# 绘制正SOC增量（蓝色）和负SOC增量（红色）数据点
sns.scatterplot(x=positive_points[lai_col], y=positive_points[soc_col],
                color="blue", alpha=0.7, label="Positive SOC Increase")
sns.scatterplot(x=negative_points[lai_col], y=negative_points[soc_col],
                color="red", alpha=0.7, label="Negative SOC Increase")

# 绘制各模型拟合曲线，并在图上标注模型公式
for name, model in models.items():
    if name in fit_results:
        params = fit_results[name]
        y_vals = model(x_vals, *params)
        plt.plot(x_vals, y_vals, label=f"{name} Fit")
        # 格式化标注公式
        if name == "Linear":
            equation_text = f"y = {params[0]:.4f} * LAI + {params[1]:.4f}"
        elif name == "Exponential":
            equation_text = f"y = {params[0]:.4f} * exp({params[1]:.4f} * LAI)"
        elif name == "Power-Law":
            equation_text = f"y = {params[0]:.4f} * LAI^{params[1]:.4f}"
        elif name == "Logarithmic":
            equation_text = f"y = {params[0]:.4f} * log(LAI) + {params[1]:.4f}"

        # 动态确定公式标注的位置（在x_vals的10百分位处标注）
        x_pos = np.percentile(x_vals, 10)
        y_pos = model(x_pos, *params) * 1.1  # 在拟合曲线上方稍微偏移
        plt.text(x_pos, y_pos, equation_text, fontsize=10, color="black",
                 bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel("2007 LAI")
plt.ylabel("SOC Monthly Increase (g/kg/month)")
plt.title("Empirical Fit of 2007 LAI vs. SOC Monthly Increase (Combined Data)")
plt.legend()
plt.grid(True)
plt.show()
