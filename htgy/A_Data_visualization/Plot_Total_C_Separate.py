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

plt.rcParams.update({
    "font.size": 16,          # base font
    "axes.titlesize": 18,     # title
    "axes.labelsize": 18,     # x/y label
    "legend.fontsize": 16,    # legend
    "xtick.labelsize": 16,    # x tick labels
    "ytick.labelsize": 16,    # y tick labels
})

# =============================================================================
# 1) Configuration & File-path Setup
# =============================================================================
past_nc     = OUTPUT_DIR / "Data" / "SOC_Past 2" / "Total_C_1950-2007_monthly.nc"
present_dir = OUTPUT_DIR / "Data" / "SOC_Present 7"
future_dir  = OUTPUT_DIR / "Data" / "SOC_Future 7"
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

# Save monthly SOC means to Excel
monthly_df   = df_all[["scenario", "date", "mean"]]
monthly_xlsx = OUTPUT_DIR / "monthly_mean_soc_by_scenario.xlsx"
monthly_df.to_excel(monthly_xlsx, index=False)
print(f"Monthly means saved to: {monthly_xlsx}")

# =============================================================================
# 3) Compute Annual Means
# =============================================================================
df_all["year"] = df_all["date"].dt.year
annual_means = df_all.groupby(["scenario", "year"], as_index=False).mean(numeric_only=True)[["scenario", "year", "mean"]]

# =============================================================================
# 4) Compute Parametric 95% CI per Scenario-Year & Save CSV
# =============================================================================
ci_records = []
for (scen, year), grp in df_all.groupby(["scenario", "year"]):
    if year < 1950 or year > 2100:
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
        "year":     int(year),
        "mean":     m,
        "lower":    m - margin,
        "upper":    m + margin
    })
ci_df = pd.DataFrame(ci_records)
ci_csv = OUTPUT_DIR / "soc_parametric95ci_by_scenario_1950_2100.csv"
ci_df.to_csv(ci_csv, index=False)
print(f"Per-scenario 95% CI saved to: {ci_csv}")

# =============================================================================
# Helper: plotting for a given date window
# =============================================================================
def plot_soc_window(df_all, annual_means, ci_df, start_date, end_date, out_path, title,
                    show_monthly=True, show_annual=True):
    fig, ax = plt.subplots(figsize=(20, 6))

    # plot order so connectors work
    scenario_order = ["Past", "Present", "ssp126", "ssp245", "ssp370", "ssp585"]

    prev_scen = None
    prev_month_date = None
    prev_month_mean = None

    for scen in scenario_order:
        df_s = df_all[df_all["scenario"] == scen].sort_values("date")
        if df_s.empty:
            continue

        # filter window
        df_s_win = df_s[(df_s["date"] >= start_date) & (df_s["date"] <= end_date)]
        if df_s_win.empty:
            continue

        dates = df_s_win["date"]
        means = df_s_win["mean"]

        col = None

        # monthly
        if show_monthly:
            line, = ax.plot(dates, means, label=scen, linewidth=0.6)
            col = line.get_color()

            # connectors (only if both ends exist in this window)
            if prev_scen == "Past" and scen == "Present":
                ax.plot([prev_month_date, dates.iloc[0]], [prev_month_mean, means.iloc[0]], color=col, linewidth=0.6)
            if prev_scen == "Present" and scen == "ssp126":
                ax.plot([prev_month_date, dates.iloc[0]], [prev_month_mean, means.iloc[0]], color=col, linewidth=0.6)

            prev_month_date = dates.iloc[-1]
            prev_month_mean = means.iloc[-1]
        else:
            # still need a color for annual+CI; derive from a dummy plot then remove
            dummy, = ax.plot([dates.iloc[0], dates.iloc[0]], [means.iloc[0], means.iloc[0]], alpha=0)
            col = dummy.get_color()
            dummy.remove()

        # annual
        if show_annual:
            df_a = annual_means[annual_means["scenario"] == scen].sort_values("year")
            df_a = df_a[(df_a["year"] >= pd.to_datetime(start_date).year) & (df_a["year"] <= pd.to_datetime(end_date).year)]
            if not df_a.empty:
                ann_dates = pd.to_datetime(df_a["year"].astype(str))
                ann_means = df_a["mean"].values
                ax.plot(ann_dates, ann_means, marker="o", markersize=4, linestyle="-", linewidth=0.8, color=col)

        # CI band
        sub = ci_df[ci_df["scenario"] == scen]
        if not sub.empty:
            sub = sub[(sub["year"] >= pd.to_datetime(start_date).year) & (sub["year"] <= pd.to_datetime(end_date).year)]
            if not sub.empty:
                ci_x = pd.to_datetime(sub["year"].astype(str))
                ax.fill_between(ci_x, sub["lower"], sub["upper"], color=col, alpha=0.35)

        prev_scen = scen

    # axis formatting
    # -------------------------------------------------------------------------
    # y-axis: auto-fit to the data in this window (monthly + annual + CI)
    # -------------------------------------------------------------------------
    y_vals = []

    # monthly values in the window
    y_vals.append(df_all.loc[(df_all["date"] >= start_date) & (df_all["date"] <= end_date), "mean"].to_numpy())

    # annual means in the window
    y_vals.append(annual_means.loc[
                      (annual_means["year"] >= pd.to_datetime(start_date).year) &
                      (annual_means["year"] <= pd.to_datetime(end_date).year),
                      "mean"
                  ].to_numpy())

    # CI bounds in the window
    y_vals.append(ci_df.loc[
                      (ci_df["year"] >= pd.to_datetime(start_date).year) &
                      (ci_df["year"] <= pd.to_datetime(end_date).year),
                      ["lower", "upper"]
                  ].to_numpy().ravel())

    y_all = np.concatenate([v for v in y_vals if v.size > 0])
    y_all = y_all[np.isfinite(y_all)]

    if y_all.size > 0:
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))

        # padding so the curve does not touch the borders
        pad = 0.08 * (y_max - y_min + 1e-12)

        ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_title(title, fontsize = 18)
    ax.set_xlabel("Year", fontsize = 18)
    ax.set_ylabel("Mean Total SOC (g/kg)", fontsize = 18)

    # ticks: decade ticks within the window
    tick_start_year = (pd.to_datetime(start_date).year // 10) * 10
    tick_end_year = ((pd.to_datetime(end_date).year + 9) // 10) * 10
    tick_years = pd.date_range(start=f"{tick_start_year}-01-01", end=f"{tick_end_year}-01-01", freq="10YS")
    ax.set_xticks(tick_years)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # pad x-limits slightly
    start = pd.to_datetime(start_date) - pd.Timedelta(days=365)
    end   = pd.to_datetime(end_date) + pd.Timedelta(days=730)
    ax.set_xlim(left=start, right=end)

    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to: {out_path}")


# =============================================================================
# 5) Make TWO figures: 1950–2024 and 2025–2100
# =============================================================================
out_path_1 = OUTPUT_DIR / "soc_timeseries_by_scenario_with_CI_1950_2024.png"
plot_soc_window(
    df_all=df_all,
    annual_means=annual_means,
    ci_df=ci_df,
    start_date="1950-01-01",
    end_date="2024-12-01",
    out_path=out_path_1,
    title="Soil Organic Carbon (Total_C) by Scenario with 95% Parametric CI (1950–2024)"
)

out_path_2 = OUTPUT_DIR / "soc_timeseries_by_scenario_with_CI_2025_2100.png"
plot_soc_window(
    df_all=df_all,
    annual_means=annual_means,
    ci_df=ci_df,
    start_date="2025-01-01",
    end_date="2100-12-01",
    out_path=out_path_2,
    title="Soil Organic Carbon (Total_C) by Scenario with 95% Parametric CI (2025–2100)"
)

# =============================================================================
# 6) Save Annual Means CSV
# =============================================================================
annual_means.to_csv(OUTPUT_DIR / "annual_mean_soc_by_scenario.csv", index=False)
print("Annual means saved to: annual_mean_soc_by_scenario.csv")
