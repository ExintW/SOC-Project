#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 1) Imports
import sys
import os
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# 把项目根目录加入搜索路径，以便 import globals
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import OUTPUT_DIR    # OUTPUT_DIR 已在 globals 中定义

# 2) Configuration & File-path Setup
past_nc         = OUTPUT_DIR / "Data" / "SOC_Past 2"   / "Total_C_1950-2007_monthly.nc"
present_dir     = OUTPUT_DIR / "Data" / "SOC_Present 6"
future_dir      = OUTPUT_DIR / "Data" / "SOC_Future 6"
future_scenarios = ["126", "245", "370", "585"]

output_csv = OUTPUT_DIR / "yearly_mean_std_total_C_by_scenario.csv"

# 3) Read Data & Compute Yearly Mean and Std Dev
records = []

# --- Past (1950–2007) ---
with xr.open_dataset(past_nc) as ds:
    da = ds["total_C"]
    # 自动找出除 time 外的所有空间维度
    spatial_dims = [d for d in da.dims if d != "time"]
    # 按月先做空间平均，结果转成 pandas.Series（索引是 Timestamp）
    monthly_series = da.mean(dim=spatial_dims, skipna=True).to_series()
    # 按年分组，计算年度均值和样本标准差（ddof=1）
    yearly = monthly_series.groupby(monthly_series.index.year).agg(['mean', 'std'])
    for year, row in yearly.iterrows():
        records.append({
            "scenario": "Past",
            "year":     int(year),
            "mean":     float(row["mean"]),
            "std":      float(row["std"])
        })

# --- Present (2008–2024) ---
for year in range(2007, 2025):
    monthly_vals = []
    for month in range(1, 13):
        path = present_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"
        df   = pd.read_parquet(path)
        monthly_vals.append(df["Total_C"].mean(skipna=True))
    arr = np.array(monthly_vals)
    records.append({
        "scenario": "Present",
        "year":     year,
        "mean":     arr.mean(),
        "std":      arr.std(ddof=1)
    })

# --- Future (2025–2100) for each SSP ---
for scen in future_scenarios:
    scen_label = f"ssp{scen}"
    scen_dir   = future_dir / scen
    for year in range(2025, 2101):
        monthly_vals = []
        for month in range(1, 13):
            path = scen_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"
            df   = pd.read_parquet(path)
            monthly_vals.append(df["Total_C"].mean(skipna=True))
        arr = np.array(monthly_vals)
        records.append({
            "scenario": scen_label,
            "year":     year,
            "mean":     arr.mean(),
            "std":      arr.std(ddof=1)
        })

# 4) Save Results
df_yearly = pd.DataFrame(records, columns=["scenario","year","mean","std"])
df_yearly.to_csv(output_csv, index=False)
print(f"Yearly mean ± std dev saved to: {output_csv}")
