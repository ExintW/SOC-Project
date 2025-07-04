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

plt.figure(figsize=(16, 8))

# keep track of the last point we drew
prev_dec_m_last = None
prev_ts_m_last = None
prev_dec_a_last = None
prev_ts_a_last = None

for i, (label, fname) in enumerate(files.items()):
    color = f"C{i}"
    ds = xr.open_dataset(output_dir / fname)
    ts_month = ds["lai"].mean(dim="points")

    # --- monthly decimal‐year vector
    times_m = ts_month["time"].values
    dec_m = np.array([t.year + (t.month - 0.5) / 12 for t in times_m])

    # if this segment starts after the last one ended, draw a tiny connector
    if prev_dec_m_last is not None and dec_m[0] > prev_dec_m_last:
        plt.plot(
            [prev_dec_m_last, dec_m[0]],
            [prev_ts_m_last, ts_month.values[0]],
            color=color, linewidth=1, alpha=0.4
        )

    # now plot the full monthly series
    plt.plot(dec_m, ts_month.values, color=color, linewidth=1, alpha=0.4)

    # --- annual mean
    ts_annual = ts_month.resample(time="YS").mean(dim="time")
    dec_a = np.array([t.year + 0.5 for t in ts_annual["time"].values])

    # connector for annual
    if prev_dec_a_last is not None and dec_a[0] > prev_dec_a_last:
        plt.plot(
            [prev_dec_a_last, dec_a[0]],
            [prev_ts_a_last, ts_annual.values[0]],
            color=color, linewidth=1, marker="o", linestyle="-"
        )

    # plot annual points+line
    plt.plot(dec_a, ts_annual.values,
             color=color, linewidth=1, marker="o", label=label)

    # record for CSV
    for t, val in zip(ts_annual["time"].values, ts_annual.values):
        records.append({
            "scenario": label,
            "year": int(t.year),
            "annual_mean_lai": float(val)
        })

    # update our “previous” markers
    prev_dec_m_last = dec_m[-1]
    prev_ts_m_last = ts_month.values[-1]
    prev_dec_a_last = dec_a[-1]
    prev_ts_a_last = ts_annual.values[-1]

    ds.close()

# finalize the figure
plt.xlabel("Year")
plt.ylabel("Spatial mean LAI")
plt.xlim(1950, 2100)
plt.title("Monthly vs. Annual Mean LAI (1950–2100)")
plt.legend(title="Scenario")
plt.tight_layout()

# --- SAVE FIGURE HERE ---
out_png = OUTPUT_DIR / "lai_monthly_vs_annual_1950-2100.png"
plt.savefig(out_png, dpi=300)
print(f"→ Saved figure to: {out_png}")

# save CSV
annual_df = pd.DataFrame.from_records(records)
csv_path = OUTPUT_DIR / "annual_mean_lai_by_scenario.csv"
annual_df.to_csv(csv_path, index=False)
print(f"→ Saved annual means CSV to {csv_path}")

plt.show()
