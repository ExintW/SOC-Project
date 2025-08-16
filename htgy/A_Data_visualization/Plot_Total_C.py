import sys
import os
import netCDF4 as nc
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from pathlib import Path
import matplotlib.dates as mdates

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *    # defines OUTPUT_DIR

# =============================================================================
# 1) Configuration & File-path Setup
# =============================================================================
past_nc     = OUTPUT_DIR / "Data" / "SOC_Past 2" / "Total_C_1950-2007_monthly.nc"
present_dir = OUTPUT_DIR / "Data" / "SOC_Present 6"
future_dir  = OUTPUT_DIR / "Data" / "SOC_Future 6"
future_scenarios = ["126", "245", "370", "585"]

# =============================================================================
# 2) Read & Aggregate Monthly Means
# =============================================================================
records = []

# Past (1950–2007)
with xr.open_dataset(past_nc) as ds:
    ds_sel = ds.sel(time=slice("1950-01-01", "2006-12-01"))
    for t_val in ds_sel.time.values:
        arr      = ds_sel["total_C"].sel(time=t_val).values
        mean_val = np.nanmean(arr)
        records.append({
            "date":     pd.Timestamp(t_val),
            "mean":     mean_val,
            "scenario": "Past"
        })

# Present (2008–2024)
for year in range(2007, 2025):
    for month in range(1, 13):
        path     = present_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"
        df       = pd.read_parquet(path)
        mean_val = df["Total_C"].mean()
        records.append({
            "date":     pd.Timestamp(year=year, month=month, day=1),
            "mean":     mean_val,
            "scenario": "Present"
        })

# Future (2025–2100) for each SSP
for scen in future_scenarios:
    scen_label = f"ssp{scen}"
    scen_dir   = future_dir / scen
    for year in range(2025, 2101):
        for month in range(1, 13):
            path     = scen_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"
            df       = pd.read_parquet(path)
            mean_val = df["Total_C"].mean()
            records.append({
                "date":     pd.Timestamp(year=year, month=month, day=1),
                "mean":     mean_val,
                "scenario": scen_label
            })

# assemble DataFrame
df_all = pd.DataFrame(records)
df_all["date"] = pd.to_datetime(df_all["date"])

# NEW: Save monthly SOC means to Excel
# ──────────────────────────────────────────────────────────────────────────────
monthly_df   = df_all[["scenario", "date", "mean"]]
monthly_xlsx = OUTPUT_DIR / "monthly_mean_soc_by_scenario.xlsx"
monthly_df.to_excel(monthly_xlsx, index=False)
print(f"Monthly means saved to: {monthly_xlsx}")

# =============================================================================
# 3) Compute Annual Means
# =============================================================================
df_all["year"] = df_all["date"].dt.year
annual_means = df_all.groupby(["scenario", "year"], as_index=False).mean()[["scenario", "year", "mean"]]

# =============================================================================
# 4) Compute Parametric 95% CI per Scenario-Year & Save CSV
# =============================================================================
ci_records = []
for (scen, year), grp in df_all.groupby(["scenario", "year"]):
    if year < 2007 or year > 2100:
        continue
    vals = grp["mean"].values
    if len(vals) < 2:
        continue
    m      = vals.mean()
    s      = vals.std(ddof=1)
    tcrit  = t.ppf(0.975, df=len(vals)-1)
    margin = tcrit * s / np.sqrt(len(vals))
    ci_records.append({
        "scenario": scen,
        "year":     year,
        "mean":     m,
        "lower":    m - margin,
        "upper":    m + margin
    })
ci_df = pd.DataFrame(ci_records)
ci_csv = OUTPUT_DIR / "soc_parametric95ci_by_scenario_2025_2100.csv"
ci_df.to_csv(ci_csv, index=False)
print(f"Per-scenario 95% CI saved to: {ci_csv}")

# =============================================================================
# 5) Plot Monthly & Annual Time Series + Scenario-Specific CI Bands + Connectors
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
    means = df_s["mean"]

    # plot monthly
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

    # plot annual
    df_a = annual_means[annual_means["scenario"] == scen].sort_values("year")
    ann_dates = pd.to_datetime(df_a["year"].astype(str))
    ann_means = df_a["mean"].values

    # connector Past → Present (annual)
    #if prev_scen == "Past" and scen == "Present":
        #ax.plot([prev_ann_date, ann_dates.iloc[0]], [prev_ann_mean, ann_means[0]], color=col, linewidth=None, marker=None)
    # connector Present → ssp126 (annual)
    #if prev_scen == "Present" and scen == "ssp126":
        #ax.plot([prev_ann_date, ann_dates.iloc[0]], [prev_ann_mean, ann_means[0]], color=col, linewidth=None, marker=None)

    ax.plot(ann_dates, ann_means, marker='o', markersize=4, linestyle='-', linewidth=0.8, color=col)

    prev_ann_date = ann_dates.iloc[-1]
    prev_ann_mean = ann_means[-1]
    prev_scen = scen

    # shade CI band for this scenario
    sub = ci_df[ci_df["scenario"] == scen]
    if not sub.empty:
        ci_x = pd.to_datetime(sub["year"].astype(str))
        ax.fill_between(ci_x, sub["lower"], sub["upper"], color=col, alpha=0.4)

# adjust axes
ax.set_xlim(left=pd.to_datetime('1950-01-01'))
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
end   = pd.to_datetime('2100-01-01') + pd.Timedelta(days=730) # or leave this off to auto‐compute the right limit
ax.set_xlim(left=start, right=end)

ax.set_title("Soil Organic Carbon (Total_C) by Scenario with 95% Parametric CI")
ax.set_xlabel("Year")
ax.set_ylabel("Mean Total_C")
ax.legend()
plt.tight_layout()

# =============================================================================
# 6) Save Figure
# =============================================================================
out_path = OUTPUT_DIR / "soc_timeseries_by_scenario_with_CI.png"
fig.savefig(out_path, dpi=300)
print(f"Figure saved to: {out_path}")

annual_means.to_csv(OUTPUT_DIR/"annual_mean_soc_by_scenario.csv", index=False)
print("Annual means saved to: annual_mean_soc_by_scenario.csv")
