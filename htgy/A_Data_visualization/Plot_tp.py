import sys, os
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── PROJECT SETUP ────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR  # Path objects

# ─── FUNCTIONS ────────────────────────────────────────────────────────────────
def compute_era5_tp(start_year, end_year):
    """Compute annual mean tp (mm/year) from ERA5 .nc files for 1950–2024."""
    tp_hist = []
    years_hist = list(range(start_year, end_year+1))
    for year in years_hist:
        nc_file = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{year}.nc"
        with nc.Dataset(nc_file) as ds:
            tp = ds.variables['tp'][:]           # (12, n_points)
            tp_mm = tp * 30 * 1000.0             # mm/month
            annual = np.sum(tp_mm, axis=0)       # (n_points,)
            tp_hist.append(np.mean(annual))
    return years_hist, tp_hist

def compute_cmip6_tp(scenario_tag, year_start=2025, year_end=2100):
    """
    Compute annual mean tp (mm/year) from a single CMIP6 scenario,
    reading pr (kg m^-2 s^-1) from 2015–2100 file and converting to mm/month.
    """
    fn = f"resampled_pr_points_2015-2100_{scenario_tag}.nc"
    ds = nc.Dataset(PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / fn)
    pr = ds.variables['pr'][:]           # (time, n_points)
    pr_mm = pr * 86400.0 * 30.0          # mm/month
    ds.close()

    years = list(range(year_start, year_end+1))
    tp_vals = []
    for i, yr in enumerate(years):
        idx0 = i * 12
        idx1 = idx0 + 12
        annual = np.sum(pr_mm[idx0:idx1, :], axis=0)
        tp_vals.append(np.mean(annual))
    return years, tp_vals

# ─── MAIN SCRIPT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Historical ERA5
    hist_years, hist_tp = compute_era5_tp(1950, 2024)
    last_year, last_tp = hist_years[-1], hist_tp[-1]

    # 2) CMIP6 scenarios
    # key = tag in filename, value = name in CSV/legend
    scenarios = {
        "126": "ssp126",
        "245": "ssp245",
        "370": "ssp370",
        "585": "ssp585",
    }
    cmip_data = {}
    for tag, name in scenarios.items():
        yrs, tp_vals = compute_cmip6_tp(tag, 2025, 2100)
        cmip_data[tag] = {"name": name, "years": yrs, "tp": tp_vals}

    # 3) Build a long‐format table
    records = []
    for year, tp in zip(hist_years, hist_tp):
        phase = "Past" if year <= 2006 else "Present"
        records.append({"scenario": phase, "year": year, "tp": tp})
    for tag, info in cmip_data.items():
        for year, tp in zip(info["years"], info["tp"]):
            records.append({"scenario": info["name"], "year": year, "tp": tp})

    df_long = pd.DataFrame(records)

    # 4) Save CSV (keep original name)
    out_csv = OUTPUT_DIR / "tp_1950-2100_mean_tp.csv"
    df_long.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved long‐format CSV to: {out_csv}")

    # 5) Plot
    plt.figure(figsize=(10, 6))

    # historical ERA5 in thick black
    plt.plot(hist_years, hist_tp,
             color='black', linewidth=1,
             label='1950–2024 (ERA5)')

    # each scenario, prepending the 2024 point so 2024→2025 is connected
    for tag, info in cmip_data.items():
        yrs_plot = [last_year] + info["years"]
        tp_plot  = [last_tp]  + info["tp"]
        plt.plot(yrs_plot, tp_plot,
                 linewidth=1, label=info["name"])

    plt.xlabel('Year')
    plt.ylabel('Annual Total Precipitation (mm)')
    plt.title('Precipitation 1950–2100: Historical + CMIP6 Scenarios')
    plt.legend()
    plt.tight_layout()

    # 6) Save figure (keep original name)
    out_png = OUTPUT_DIR / "tp_1950-2100_era5_cmip6_scenarios.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved figure to: {out_png}")
    plt.show()
