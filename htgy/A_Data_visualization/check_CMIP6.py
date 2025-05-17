import xarray as xr
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from globals import *  # Expects DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

# 重新打开数据并计算区域平均
ds = xr.open_dataset("LAI_LoessPlateau_195001-201412.nc")
lai_regional = ds.lai.mean(dim=["lat", "lon"])

# 转成 Pandas Series
ts = lai_regional.to_pandas()

# 将 cftime 索引转成标准时间
ts.index = pd.to_datetime(ts.index.astype(str))

year_frac = ts.index.year + (ts.index.month - 0.5) / 12

# --- 3. 用 polyfit 拟合一次多项式 (线性趋势) ---
#    z[0] 是斜率 (unit: LAI per year)，z[1] 是截距
z = np.polyfit(year_frac, ts.values, 1)
trend = np.polyval(z, year_frac)

# --- 4. 画图：原始月度 + 趋势线 ---
plt.figure(figsize=(12, 4))
plt.plot(ts.index, ts.values,
         color='blue',      # 深蓝色
         linewidth=1.0,    
         alpha=1.0,         
         label='Monthly Mean LAI')
plt.plot(ts.index, trend,
         color='red', linestyle='--', linewidth=2,
         label=f'Trend: {z[0]:.4f} LAI/year')

plt.title("Loess Plateau Monthly Mean LAI (1950–2014) with Trend")
plt.xlabel("Time")
plt.ylabel("LAI")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ts.to_csv(OUTPUT_DIR / "LAI_LoessPlateau_monthly_195001-201412.csv", header=["LAI"])