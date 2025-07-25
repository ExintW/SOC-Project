import sys
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *    # defines OUTPUT_DIR

# =============================================================================
# 1) Configuration & File-path Setup
# =============================================================================
past_nc = OUTPUT_DIR / "Total_C_1950-2007_monthly.nc"

# =============================================================================
# 2) Read & Aggregate Monthly Means (1950–2007 only)
# =============================================================================
records = []
with xr.open_dataset(past_nc) as ds:
    ds_sel = ds.sel(time=slice("1950-01-01", "2007-12-01"))
    for t_val in ds_sel.time.values:
        arr = ds_sel["total_C"].sel(time=t_val).values
        mean_val = np.nanmean(arr)
        records.append({
            "date": pd.Timestamp(t_val),
            "mean": mean_val,
            "scenario": "Past"
        })

df_all = pd.DataFrame(records)
df_all["date"] = pd.to_datetime(df_all["date"])
df_all["year"] = df_all["date"].dt.year

# Compute annual means
annual_means = df_all.groupby("year", as_index=False)["mean"].mean()

# =============================================================================
# 3) Plot Past Monthly & Annual Time Series
# =============================================================================
fig, ax = plt.subplots(figsize=(20, 6))

# Monthly
ax.plot(df_all["date"], df_all["mean"], label="Monthly (Past)", linewidth=0.6)

# Annual
ann_dates = pd.to_datetime(annual_means["year"].astype(str))
ann_means = annual_means["mean"].values
ax.plot(ann_dates, ann_means, marker='o', markersize=4, linestyle='-', linewidth=0.8, label="Annual (Past)")

# Axes formatting
tick_years = pd.date_range(start='1950-01-01', end='2010-01-01', freq='10YS')
ax.set_xticks(tick_years)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlim(pd.to_datetime('1949-01-01'), pd.to_datetime('2010-01-01'))
ax.set_ylim(bottom=0)

ax.set_title("Soil Organic Carbon (Total_C) 1950–2007")
ax.set_xlabel("Year")
ax.set_ylabel("Mean Total_C")
ax.legend()
plt.tight_layout()

# =============================================================================
# 4) Save Figure
# =============================================================================
out_path = OUTPUT_DIR / "soc_past_timeseries.png"
fig.savefig(out_path, dpi=300)
print(f"Figure saved to: {out_path}")
