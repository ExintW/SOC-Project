import sys
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.patches import Patch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # defines OUTPUT_DIR

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# =============================================================================
# 1) Configuration and File-path Setup
# =============================================================================
past_nc     = OUTPUT_DIR / "Data" / "SOC_Past 2" / "Total_C_1950-2007_monthly.nc"
present_dir = OUTPUT_DIR / "Data" / "SOC_Present 7"
future_dir  = OUTPUT_DIR / "Data" / "SOC_Future 7"
future_scenarios = ["126", "245", "370", "585"]

# Change points for the first figure
CHANGE_SEGMENTS_1950_2024 = [
    (1950, 1973),
    (1973, 2000),
    (2000, 2024),
]

# =============================================================================
# 2) Read and Aggregate Monthly Means
# =============================================================================
records = []

# Past (1950–2006)
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

# Present (2007–2024)
for year in range(2007, 2025):
    for month in range(1, 13):
        path = present_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"
        df = pd.read_parquet(path)
        mean_val = df["Total_C"].mean()
        records.append({
            "date":     pd.Timestamp(year=year, month=month, day=1),
            "mean":     mean_val,
            "scenario": "Present"
        })

# Future (2025–2100) for each SSP
for scen in future_scenarios:
    scen_label = f"ssp{scen}"
    scen_dir = future_dir / scen
    for year in range(2025, 2101):
        for month in range(1, 13):
            path = scen_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"
            df = pd.read_parquet(path)
            mean_val = df["Total_C"].mean()
            records.append({
                "date":     pd.Timestamp(year=year, month=month, day=1),
                "mean":     mean_val,
                "scenario": scen_label
            })

# assemble DataFrame
df_all = pd.DataFrame(records)
df_all["date"] = pd.to_datetime(df_all["date"])

# Save monthly SOC means to Excel (optional output)
monthly_df = df_all[["scenario", "date", "mean"]]
monthly_xlsx = OUTPUT_DIR / "monthly_mean_soc_by_scenario.xlsx"
monthly_df.to_excel(monthly_xlsx, index=False)
print(f"Monthly means saved to: {monthly_xlsx}")

# =============================================================================
# 3) Compute Annual Means
# =============================================================================
df_all["year"] = df_all["date"].dt.year
annual_means = (
    df_all.groupby(["scenario", "year"], as_index=False)
    .mean(numeric_only=True)[["scenario", "year", "mean"]]
)

# =============================================================================
# 4) Compute Parametric 95% CI per Scenario-Year and Save CSV
# =============================================================================
ci_records = []
for (scen, year), grp in df_all.groupby(["scenario", "year"]):
    if year < 1950 or year > 2100:
        continue

    vals = grp["mean"].values
    if len(vals) < 2:
        continue

    m = vals.mean()
    s = vals.std(ddof=1)
    tcrit = t.ppf(0.975, df=len(vals) - 1)
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
# Helper 1: annual-only plotting for future (2025–2100), scenario colored
# =============================================================================
def plot_soc_window_annual_only(annual_means, ci_df, start_date, end_date, out_path, title):
    fig, ax = plt.subplots(figsize=(20, 6))

    scenario_order = ["ssp126", "ssp245", "ssp370", "ssp585"]

    for scen in scenario_order:
        df_a = annual_means[annual_means["scenario"] == scen].sort_values("year")
        df_a = df_a[
            (df_a["year"] >= pd.to_datetime(start_date).year) &
            (df_a["year"] <= pd.to_datetime(end_date).year)
        ]
        if df_a.empty:
            continue

        ann_dates = pd.to_datetime(df_a["year"].astype(str))
        ann_means = df_a["mean"].values

        line, = ax.plot(ann_dates, ann_means, marker="o", markersize=4, linewidth=1.2, label=scen)
        col = line.get_color()

        sub = ci_df[ci_df["scenario"] == scen]
        sub = sub[
            (sub["year"] >= pd.to_datetime(start_date).year) &
            (sub["year"] <= pd.to_datetime(end_date).year)
        ]
        if not sub.empty:
            ci_x = pd.to_datetime(sub["year"].astype(str))
            ax.fill_between(ci_x, sub["lower"], sub["upper"], color=col, alpha=0.25)

    # y-axis auto-fit
    y_vals = []
    y_vals.append(annual_means.loc[
        (annual_means["year"] >= pd.to_datetime(start_date).year) &
        (annual_means["year"] <= pd.to_datetime(end_date).year),
        "mean"
    ].to_numpy())

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
        pad = 0.08 * (y_max - y_min + 1e-12)
        ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Mean Total SOC (g/kg)", fontsize=18)

    # decade ticks
    tick_start_year = (pd.to_datetime(start_date).year // 10) * 10
    tick_end_year = ((pd.to_datetime(end_date).year + 9) // 10) * 10
    tick_years = pd.date_range(start=f"{tick_start_year}-01-01", end=f"{tick_end_year}-01-01", freq="10YS")
    ax.set_xticks(tick_years)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    start = pd.to_datetime(start_date) - pd.Timedelta(days=365)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=730)
    ax.set_xlim(left=start, right=end)

    # --- Add CI legend proxy (ONE entry for all shaded areas) ---
    ci_patch = Patch(facecolor="gray", alpha=0.25, label="95% CI")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(ci_patch)
    labels.append("95% CI")
    ax.legend(handles, labels)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to: {out_path}")

# =============================================================================
# Helper 2: annual-only plotting for 1950–2024, colored by change-point segments
# =============================================================================
def plot_soc_annual_change_segments_1950_2024(annual_means, ci_df, segments, out_path, title):
    """
    Plots ONE continuous annual SOC series for 1950–2024 (Past + Present combined),
    and colors it by the change-point segments:
      1950–1975, 1975–2000, 2000–2024
    Also keeps the 95% CI shading, using the same segment colors.
    """
    fig, ax = plt.subplots(figsize=(20, 6))

    # Build combined annual series (Past + Present)
    df_past = annual_means[annual_means["scenario"] == "Past"].copy()
    df_pres = annual_means[annual_means["scenario"] == "Present"].copy()

    df_comb = pd.concat([df_past, df_pres], ignore_index=True)
    df_comb = df_comb.sort_values("year")
    df_comb = df_comb[(df_comb["year"] >= 1950) & (df_comb["year"] <= 2024)]

    # Combine CI as well
    ci_past = ci_df[ci_df["scenario"] == "Past"].copy()
    ci_pres = ci_df[ci_df["scenario"] == "Present"].copy()
    ci_comb = pd.concat([ci_past, ci_pres], ignore_index=True)
    ci_comb = ci_comb.sort_values("year")
    ci_comb = ci_comb[(ci_comb["year"] >= 1950) & (ci_comb["year"] <= 2024)]

    # Get matplotlib default cycle colors
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_iter = iter(cycle_colors)

    # Plot each segment as its own colored line
    for (y0, y1) in segments:
        col = next(color_iter)

        seg = df_comb[(df_comb["year"] >= y0) & (df_comb["year"] <= y1)]
        if seg.empty:
            continue

        x = pd.to_datetime(seg["year"].astype(str))
        y = seg["mean"].values

        ax.plot(x, y, marker="o", markersize=4, linewidth=1.4, color=col, label=f"{y0}–{y1}")

        # CI shading for the same segment
        seg_ci = ci_comb[(ci_comb["year"] >= y0) & (ci_comb["year"] <= y1)]
        if not seg_ci.empty:
            x_ci = pd.to_datetime(seg_ci["year"].astype(str))
            ax.fill_between(x_ci, seg_ci["lower"], seg_ci["upper"], color=col, alpha=0.25)

    # y-axis auto-fit using combined annual + CI
    y_vals = []
    y_vals.append(df_comb["mean"].to_numpy())
    y_vals.append(ci_comb[["lower", "upper"]].to_numpy().ravel())

    y_all = np.concatenate([v for v in y_vals if v.size > 0])
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size > 0:
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        pad = 0.08 * (y_max - y_min + 1e-12)
        ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Mean Total SOC (g/kg)", fontsize=18)

    # decade ticks
    tick_start_year = 1950
    tick_end_year = 2030
    tick_years = pd.date_range(start=f"{tick_start_year}-01-01", end=f"{tick_end_year}-01-01", freq="10YS")
    ax.set_xticks(tick_years)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # pad x-limits slightly
    start = pd.to_datetime("1950-01-01") - pd.Timedelta(days=365)
    end = pd.to_datetime("2024-12-01") + pd.Timedelta(days=730)
    ax.set_xlim(left=start, right=end)

    # --- Add CI legend proxy (ONE entry for all shaded areas) ---
    ci_patch = Patch(facecolor="gray", alpha=0.25, label="95% CI")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(ci_patch)
    labels.append("95% CI")
    ax.legend(handles, labels)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to: {out_path}")

# =============================================================================
# 5) Make TWO figures: 1950–2024 and 2025–2100
# =============================================================================

# First figure: annual-only, colored by change-point segments
out_path_1 = OUTPUT_DIR / "soc_annual_change_segments_1950_2024.png"
plot_soc_annual_change_segments_1950_2024(
    annual_means=annual_means,
    ci_df=ci_df,
    segments=CHANGE_SEGMENTS_1950_2024,
    out_path=out_path_1,
    title="Annual Mean Soil Organic Carbon with 95% CI (1950-2024)"
)

# Second figure: annual-only, scenario colored (future)
out_path_2 = OUTPUT_DIR / "soc_annual_by_scenario_with_CI_2025_2100.png"
plot_soc_window_annual_only(
    annual_means=annual_means,
    ci_df=ci_df,
    start_date="2025-01-01",
    end_date="2100-12-01",
    out_path=out_path_2,
    title="Annual Mean SOC by Scenario with 95% CI (2025–2100)"
)

# =============================================================================
# 6) Save Annual Means CSV
# =============================================================================
annual_csv = OUTPUT_DIR / "annual_mean_soc_by_scenario.csv"
annual_means.to_csv(annual_csv, index=False)
print(f"Annual means saved to: {annual_csv}")
