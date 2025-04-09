import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # 假设这里面定义了 OUTPUT_DIR, DATA_DIR 等

# ========================
# 1. 逐月读取并汇总模型计算结果
# ========================
total_erosion_model_2007 = 0.0

# 假设文件命名: SOC_terms_2007_{month:02d}_timestep_{month}_River.csv
for month in range(1, 13):
    month_str = f"{month:02d}"
    filename = f"SOC_terms_2007_{month_str}_timestep_{month}_River.csv"
    
    # 读取 CSV
    df_model = pd.read_csv(OUTPUT_DIR / "Data" / filename)
    
    # 对当月的 E_t_ha_month 做汇总
    month_erosion_sum = df_model["E_t_ha_month"].sum()
    total_erosion_model_2007 += month_erosion_sum

print("=== 模型计算结果：2007年全年的土壤侵蚀量 (逐月累加) ===")
print(f"累计侵蚀量: {total_erosion_model_2007:.2f} (t/ha) 或对应单位")


# ========================
# 2. 读取并提取潼关径流泥沙数据 (按年)
# ========================
df_tongguan = pd.read_excel(DATA_DIR / "潼关径流泥沙.xlsx", sheet_name="Sheet1")
print("\n潼关泥沙数据前几行:")
print(df_tongguan.head())

# 假设表格中“年份”这一列就是数值或字符串，直接可以用 == 2007 筛选
# 如果列中是日期，需要先转成 datetime 后再用 dt.year == 2007
df_tongguan_2007 = df_tongguan.loc[df_tongguan["年份"] == 2007]

# 从中取出全年泥沙量 (假设列名为 “泥沙(10^8 t)”)
# 如果每年只一条记录, 直接取 .iloc[0] 即可; 若存在多条, 可以视情况求和/平均
tongguan_sediment_2007 = df_tongguan_2007["泥沙(10^8 t)"].sum()

print("\n=== 潼关实测结果：2007年全年泥沙量 (按年份汇总) ===")
print(f"泥沙量: {tongguan_sediment_2007} (10^8 t) (示例)")


# ========================
# 3. 简单对比
# ========================
print("\n模型结果 vs 实测结果 (按全年计算, 简单示例):")
print(f"模型侵蚀总量（2007年1-12月）: {total_erosion_model_2007:.2f}")
print(f"潼关泥沙总量（2007年）: {tongguan_sediment_2007} (单位请注意换算)")

# 如需进一步做统计分析 (相关系数 / RMSE / NSE等)，
# 可以将逐月模型值 vs 逐月实测值对应起来，再计算。此处仅演示年度汇总对比。
