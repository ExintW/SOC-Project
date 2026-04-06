import os
import sys
from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── adjust this to your project structure ─────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# ─────────────────────────────────────────────────────────────────────────────

# Define output directory
output_dir = Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled"
output_dir.mkdir(parents=True, exist_ok=True)

# Filenames & labels
files = {
    "Historical (1950–2000)": "resampled_lai_points_1950-2000.nc",
    "Present (2001–2014)": "resampled_lai_points_2001-2014.nc",
    "ssp126 (2015–2100)": "resampled_lai_points_2015-2100_126.nc",
    "ssp245 (2015–2100)": "resampled_lai_points_2015-2100_245.nc",
    "ssp370 (2015–2100)": "resampled_lai_points_2015-2100_370.nc",
    "ssp585 (2015–2100)": "resampled_lai_points_2015-2100_585.nc",
}

# collector for annual CSV
records = []

# collector for historical + present bar chart (1950–2024)
hist_present_records = []

plt.figure(figsize=(16, 8))

# keep track of the last point we drew
prev_dec_m_last = None
prev_ts_m_last = None
prev_dec_a_last = None
prev_ts_a_last = None

for i, (label, fname) in enumerate(files.items()):
    color = f"C{i}"
    ds = xr.open_dataset(output_dir / fname)

    # -------------------------------------------------------------------------
    # monthly spatial mean across all points, for the line plot
    # -------------------------------------------------------------------------
    ts_month = ds["lai"].mean(dim="points")

    times_m = ts_month["time"].values
    dec_m = np.array([t.year + (t.month - 0.5) / 12 for t in times_m])

    # draw connector if there is a gap
    if prev_dec_m_last is not None and dec_m[0] > prev_dec_m_last:
        plt.plot(
            [prev_dec_m_last, dec_m[0]],
            [prev_ts_m_last, ts_month.values[0]],
            color=color,
            linewidth=1,
            alpha=0.4
        )

    # plot monthly series
    plt.plot(dec_m, ts_month.values, color=color, linewidth=1, alpha=0.4)

    # -------------------------------------------------------------------------
    # annual mean map at each point
    # then annual mean across points + annual spatial std across points
    # -------------------------------------------------------------------------
    lai_annual_map = ds["lai"].groupby("time.year").mean(dim="time")   # dims: year, points
    years = lai_annual_map["year"].values

    annual_mean_vals = lai_annual_map.mean(dim="points").values
    annual_spatial_std_vals = lai_annual_map.std(dim="points").values

    dec_a = np.array([int(y) + 0.5 for y in years])

    # connector for annual series
    if prev_dec_a_last is not None and dec_a[0] > prev_dec_a_last:
        plt.plot(
            [prev_dec_a_last, dec_a[0]],
            [prev_ts_a_last, annual_mean_vals[0]],
            color=color,
            linewidth=1,
            marker="o",
            linestyle="-"
        )

    # annual points + line
    plt.plot(
        dec_a,
        annual_mean_vals,
        color=color,
        linewidth=1,
        marker="o",
        label=label
    )

    # record for CSV
    for year, mean_val, std_val in zip(years, annual_mean_vals, annual_spatial_std_vals):
        record = {
            "scenario": label,
            "year": int(year),
            "annual_mean_lai": float(mean_val),
            "annual_spatial_std_lai": float(std_val)
        }
        records.append(record)

        # collect only 1950–2024 for bar chart
        if 1950 <= int(year) <= 2024:
            hist_present_records.append(record)

    # update previous markers
    prev_dec_m_last = dec_m[-1]
    prev_ts_m_last = ts_month.values[-1]
    prev_dec_a_last = dec_a[-1]
    prev_ts_a_last = annual_mean_vals[-1]

    ds.close()

# finalize the line figure
plt.xlabel("Year")
plt.ylabel("Spatial mean LAI")
plt.xlim(1950, 2100)
plt.title("Monthly vs. Annual Mean LAI (1950–2100)")
plt.legend(title="Scenario")
plt.tight_layout()

# --- SAVE LINE FIGURE ---
out_png = OUTPUT_DIR / "lai_monthly_vs_annual_1950-2100.png"
plt.savefig(out_png, dpi=300)
print(f"→ Saved figure to: {out_png}")

# save CSV with spatial std
annual_df = pd.DataFrame.from_records(records)
csv_path = OUTPUT_DIR / "annual_mean_spatialstd_lai_by_scenario.csv"
annual_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"→ Saved annual mean + spatial std CSV to {csv_path}")

plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Bar chart for 1950–2024 with spatial std error bars
# ─────────────────────────────────────────────────────────────────────────────
hist_present_df = pd.DataFrame.from_records(hist_present_records)
hist_present_df = hist_present_df.sort_values("year")

plt.figure(figsize=(18, 8))
plt.bar(
    hist_present_df["year"],
    hist_present_df["annual_mean_lai"],
    yerr=hist_present_df["annual_spatial_std_lai"],
    capsize=2
)

plt.xlabel("Year", fontsize=16)
plt.ylabel("Annual mean LAI", fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.title("Annual Mean LAI with Spatial Standard Deviation (1950–2024)", fontsize=16)
plt.xlim(1949.5, 2024.5)
plt.tight_layout()

bar_png = OUTPUT_DIR / "annual_mean_lai_with_spatialstd_1950-2024_bar.png"
plt.savefig(bar_png, dpi=300)
print(f"→ Saved bar chart to: {bar_png}")

plt.show()