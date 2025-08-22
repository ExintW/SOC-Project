#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute yearly spatial standard deviation of Soil Organic Carbon (Total_C)
for Past, Present, and Future SSP scenarios, and save results to CSV.
"""

import sys
import os
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# make sure your project root is on PYTHONPATH so globals can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import OUTPUT_DIR    # OUTPUT_DIR should be defined in globals.py

# 1) Configuration & File-path Setup
past_nc          = OUTPUT_DIR / "Data" / "SOC_Past 2"   / "Total_C_1950-2007_monthly.nc"
present_dir      = OUTPUT_DIR / "Data" / "SOC_Present 7"
future_dir       = OUTPUT_DIR / "Data" / "SOC_Future 7"
future_scenarios = ["126", "245", "370", "585"]

output_csv = OUTPUT_DIR / "yearly_spatial_std_total_C_by_scenario.csv"

# 2) Calculate Yearly Spatial Standard Deviation
records = []

# --- Past (1950–2006) from NetCDF ---
with xr.open_dataset(past_nc) as ds:
    # select up through December 2006 so we don't overlap with Present
    ds_sel = ds.sel(time=slice("1950-01-01", "2006-12-01"))
    da     = ds_sel["total_C"]
    # all dims except time are spatial
    spatial_dims = [d for d in da.dims if d != "time"]
    # compute per‐month spatial std dev
    monthly_spatial_std = da.std(dim=spatial_dims, skipna=True)
    # convert to pandas Series indexed by Timestamp
    monthly_std_series = monthly_spatial_std.to_series()
    # group by calendar year and take the mean of the 12 monthly spatial stds
    yearly_spatial_std  = monthly_std_series.groupby(monthly_std_series.index.year).mean()
    for year, std_val in yearly_spatial_std.items():
        records.append({
            "scenario":   "Past",
            "year":       int(year),
            "spatial_std": float(std_val)
        })

# --- Present (2007–2024) from Parquet ---
for year in range(2007, 2025):
    monthly_stds = []
    for month in range(1, 13):
        path = present_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"
        df   = pd.read_parquet(path)
        # spatial std across all grid cells for this month
        monthly_stds.append(df["Total_C"].std(ddof=1, skipna=True))
    # average the 12 monthly spatial stds to get annual spatial std
    records.append({
        "scenario":   "Present",
        "year":       year,
        "spatial_std": float(np.nanmean(monthly_stds))
    })

# --- Future (2025–2100) for each SSP scenario ---
for scen in future_scenarios:
    scen_label = f"ssp{scen}"
    scen_dir   = future_dir / scen
    for year in range(2025, 2101):
        monthly_stds = []
        for month in range(1, 13):
            path = scen_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"
            df   = pd.read_parquet(path)
            monthly_stds.append(df["Total_C"].std(ddof=1, skipna=True))
        records.append({
            "scenario":   scen_label,
            "year":       year,
            "spatial_std": float(np.nanmean(monthly_stds))
        })

# 3) Save to CSV
df_out = pd.DataFrame(records, columns=["scenario", "year", "spatial_std"])
df_out.to_csv(output_csv, index=False)
print(f"Yearly spatial standard deviation saved to: {output_csv}")
