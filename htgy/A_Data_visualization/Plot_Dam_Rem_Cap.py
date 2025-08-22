import sys
import os
import netCDF4 as nc
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *    # defines OUTPUT_DIR

# =============================================================================
# 1) Configuration & File-path Setup
# =============================================================================
past_nc     = OUTPUT_DIR / "Data" / "SOC_Past 2" / "Total_C_1950-2007_monthly.nc"  # expects 'dam_rem_cap' var
present_dir = OUTPUT_DIR / "Data" / "SOC_Present 7"
future_dir  = OUTPUT_DIR / "Data" / "SOC_Future 7"
future_scenarios = ["126", "245", "370", "585"]

# =============================================================================
# 2) Read & Aggregate Monthly Sums (dam_rem_cap)
# =============================================================================
records = []

# Past (1950–2006) from NetCDF
with xr.open_dataset(past_nc) as ds:
    ds_sel = ds.sel(time=slice("1950-01-01", "2006-12-01"))
    for t_val in ds_sel.time.values:
        # monthly spatial SUM of dam_rem_cap
        arr      = ds_sel["dam_rem_cap"].sel(time=t_val).values
        sum_val  = np.nansum(arr)
        records.append({
            "date":     pd.Timestamp(t_val),
            "mean":     sum_val,      # keep column name 'mean' to preserve downstream structure
            "scenario": "Past"
        })

# Present (2007–2024) from Parquet
for year in range(2007, 2025):
    for month in range(1, 13):
        path = present_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"
        df   = pd.read_parquet(path)
        # monthly spatial SUM of dam_rem_cap
        sum_val = df["dam_rem_cap"].sum()
        records.append({
            "date":     pd.Timestamp(year=year, month=month, day=1),
            "mean":     sum_val,      # keep name 'mean' (represents monthly sum here)
            "scenario": "Present"
        })

# Future (2025–2100) for each SSP from Parquet
for scen in future_scenarios:
    scen_label = f"ssp{scen}"
    scen_dir   = future_dir / scen
    for year in range(2025, 2101):
        for month in range(1, 13):
            path = scen_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"
            df   = pd.read_parquet(path)
            # monthly spatial SUM of dam_rem_cap
            sum_val = df["dam_rem_cap"].sum()
            records.append({
                "date":     pd.Timestamp(year=year, month=month, day=1),
                "mean":     sum_val,   # keep name 'mean'
                "scenario": scen_label
            })

# assemble DataFrame
df_all = pd.DataFrame(records)
df_all["date"] = pd.to_datetime(df_all["date"])

# Save monthly sums to Excel
monthly_df   = df_all[["scenario", "date", "mean"]]  # 'mean' holds monthly SUMs here
monthly_xlsx = OUTPUT_DIR / "monthly_sum_dam_rem_cap_by_scenario.xlsx"
monthly_df.to_excel(monthly_xlsx, index=False)
print(f"Monthly dam_rem_cap sums saved to: {monthly_xlsx}")

# =============================================================================
# 3) Compute Annual Means (average of monthly sums per year)
# =============================================================================
df_all["year"] = df_all["date"].dt.year
annual_means = (
    df_all
    .groupby(["scenario", "year"], as_index=False)
    .mean()[["scenario", "year", "mean"]]  # 'mean' now = annual mean of monthly sums
)

# =============================================================================
# 4) Plot Monthly & Annual Time Series + Connectors (no CI)
# =============================================================================
fig, ax = plt.subplots(figsize=(20, 6))
prev_scen = None
prev_month_date = None
prev_month_mean = None
prev_ann_date = None
prev_ann_mean = None

for scen, df_s in df_all.groupby("scenario"):
    df_s = df_s.sort_values("date")
    dates = df_s["date"]
    means = df_s["mean"]  # monthly sums

    # plot monthly sums
    line, = ax.plot(dates, means, label=scen, linewidth=0.6)
    col = line.get_color()

    # connector Past → Present (monthly)
    if prev_scen == "Past" and scen == "Present":
        ax.plot([prev_month_date, dates.iloc[0]], [prev_month_mean, means.iloc[0]], color=col, linewidth=0.6)
    # connector Present → ssp126 (monthly)
    if prev_scen == "Present" and scen == "ssp126":
        ax.plot([prev_month_date, dates.iloc[0]], [prev_month_mean, means.iloc[0]], color=col, linewidth=0.6)

    prev_month_date = dates.iloc[-1]
    prev_month_mean = means.iloc[-1]

    # plot annual means (average of monthly sums)
    df_a = annual_means[annual_means["scenario"] == scen].sort_values("year")
    ann_dates = pd.to_datetime(df_a["year"].astype(str))
    ann_means = df_a["mean"].values

    ax.plot(ann_dates, ann_means, marker='o', markersize=4, linestyle='-', linewidth=0.8, color=col)

    prev_ann_date = ann_dates.iloc[-1]
    prev_ann_mean = ann_means[-1]
    prev_scen = scen

# adjust axes
ax.set_ylim(bottom=0)

# 1) Make sure the axis really starts at 1950-01-01
ax.set_xlim(left=pd.to_datetime('1950-01-01'))

# 2) Generate an array of tick‐positions at each decade from 1950 to 2100
tick_years = pd.date_range(start='1950-01-01',
                           end='2100-01-01',
                           freq='10YS')   # 'YS' = year start

# 3) Apply those positions as major ticks
ax.set_xticks(tick_years)

# 4) Format them to show only the year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# 5) now extend the axis start back by 1 year (or however much you like)
start = pd.to_datetime('1950-01-01') - pd.Timedelta(days=365)
end   = pd.to_datetime('2100-01-01') + pd.Timedelta(days=730)
ax.set_xlim(left=start, right=end)

ax.set_title("dam_rem_cap: Monthly Spatial Sum (by Scenario) and Annual Means")
ax.set_xlabel("Year")
ax.set_ylabel("Monthly sum of dam_rem_cap (units)")
ax.legend()
plt.tight_layout()

# =============================================================================
# 5) Save Figure & Annual CSV
# =============================================================================
out_path = OUTPUT_DIR / "dam_rem_cap_timeseries_by_scenario.png"
fig.savefig(out_path, dpi=300)
print(f"Figure saved to: {out_path}")

annual_csv = OUTPUT_DIR / "annual_mean_dam_rem_cap_by_scenario.csv"
annual_means.to_csv(annual_csv, index=False)
print(f"Annual means saved to: {annual_csv}")
