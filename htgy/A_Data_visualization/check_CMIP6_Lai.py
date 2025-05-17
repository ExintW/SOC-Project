import os
import sys
from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── adjust this to your project structure ─────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR
# ─────────────────────────────────────────────────────────────────────────────

# Define output directory
output_dir = Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled"
output_dir.mkdir(parents=True, exist_ok=True)

# Filenames & labels
files = {
    "Historical (1950–2015)": "resampled_lai_points_1950-2015.nc",
    "ssp126 (2015–2100)":     "resampled_lai_points_2015-2100_126.nc",
    "ssp245 (2015–2100)":     "resampled_lai_points_2015-2100_245.nc",
    "ssp585 (2015–2100)":     "resampled_lai_points_2015-2100_585.nc",
}

# Prepare collector for annual means
records = []

plt.figure(figsize=(12, 6))

for i, (label, fname) in enumerate(files.items()):
    color = f"C{i}"
    ds = xr.open_dataset(output_dir / fname)
    ts_month = ds["lai"].mean(dim="points")

    # Monthly decimal‐year
    times_m = ts_month["time"].values
    dec_m = np.array([t.year + (t.month - 0.5)/12 for t in times_m])
    plt.plot(dec_m, ts_month.values, color=color, linewidth=1, alpha=0.4)

    # Annual means (using "YS" for year‐start)
    ts_annual = ts_month.resample(time="YS").mean(dim="time")
    for t, val in zip(ts_annual["time"].values, ts_annual.values):
        records.append({"scenario": label,
                        "year": int(t.year),
                        "annual_mean_lai": float(val)})

    dec_a = np.array([t.year + 0.5 for t in ts_annual["time"].values])
    plt.plot(dec_a, ts_annual.values,
             color=color, linewidth=1.5, marker="o", label=label)

    ds.close()

plt.xlabel("Year")
plt.ylabel("Spatial mean LAI")
plt.xlim(1950, 2100)
plt.title("Monthly vs. Annual Mean LAI (1950–2100)")
plt.legend(title="Scenario")
plt.tight_layout()

# Save annual means CSV into the same output_dir
annual_df = pd.DataFrame.from_records(records)
csv_path = output_dir / "annual_mean_lai_by_scenario.csv"
annual_df.to_csv(csv_path, index=False)
print(f"→ Saved annual means CSV to {csv_path}")

plt.show()
