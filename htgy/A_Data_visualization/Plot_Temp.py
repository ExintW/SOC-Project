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
def compute_era5_stl(start_year, end_year):
    """Compute annual mean STL from ERA5 using average of stl1 and stl2 (in K)."""
    stl_hist = []
    years = list(range(start_year, end_year + 1))

    for year in years:
        file_path = PROCESSED_DIR / "ERA5_Data_Temp_Monthly_Resampled" / f"resampled_{year}.nc"
        with nc.Dataset(file_path) as ds:
            stl1 = ds.variables['stl1'][:]  # shape: (12, N)
            stl2 = ds.variables['stl2'][:]
            stl_avg = (stl1 + stl2) / 2.0   # shape: (12, N)
            stl_annual = np.mean(stl_avg, axis=0)  # average across months
            stl_hist.append(np.mean(stl_annual))  # average across grid
    return years, stl_hist

def compute_cmip6_tsl(scenario_tag, year_start=2025, year_end=2100):
    """
    Compute annual mean TSL from CMIP6 using pre-averaged top 5 depth data.
    Assumes file: resampled_tsl_top5mean_points_ssp{scenario}.nc
    """
    fn = f"resampled_tsl_surface_points_ssp{scenario_tag}_2015-2100.nc"
    ds = nc.Dataset(PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / fn)
    tsl_avg_5 = ds.variables['tsl'][:]  # shape: (time, n_points)
    ds.close()

    years = list(range(year_start, year_end + 1))
    stl_vals = []
    for i, yr in enumerate(years):
        idx0 = i * 12
        idx1 = idx0 + 12
        annual = np.mean(tsl_avg_5[idx0:idx1, :], axis=0)
        stl_vals.append(np.mean(annual))  # mean over all points
    return years, stl_vals

# ─── MAIN SCRIPT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Historical ERA5
    hist_years, hist_stl = compute_era5_stl(1950, 2024)
    last_year, last_stl = hist_years[-1], hist_stl[-1]

    # 2) CMIP6 scenarios
    scenarios = {
        "126": "ssp126",
        "245": "ssp245",
        "370": "ssp370",
        "585": "ssp585",
    }
    cmip_data = {}
    for tag, name in scenarios.items():
        yrs, stl_vals = compute_cmip6_tsl(tag, 2025, 2100)
        cmip_data[tag] = {"name": name, "years": yrs, "stl": stl_vals}

    # 3) Combine into long-format DataFrame
    records = []
    for year, stl in zip(hist_years, hist_stl):
        phase = "Past" if year <= 2006 else "Present"
        records.append({"scenario": phase, "year": year, "stl": stl})
    for tag, info in cmip_data.items():
        for year, stl in zip(info["years"], info["stl"]):
            records.append({"scenario": info["name"], "year": year, "stl": stl})

    df_long = pd.DataFrame(records)

    # 4) Save CSV
    out_csv = OUTPUT_DIR / "stl_1950-2100_mean_temperature.csv"
    df_long.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved STL CSV to: {out_csv}")

    # 5) Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(hist_years, hist_stl,
             color='black', linewidth=1,
             label='1950–2024 (ERA5)')

    for tag, info in cmip_data.items():
        yrs_plot = [last_year] + info["years"]
        stl_plot = [last_stl] + info["stl"]
        plt.plot(yrs_plot, stl_plot, linewidth=1, label=info["name"])

    plt.xlabel("Year")
    plt.ylabel("Mean Soil Temperature (K)")
    plt.title("Soil Temperature 1950–2100: ERA5 and CMIP6 (Surface–Shallow)")
    plt.legend()
    plt.tight_layout()

    out_png = OUTPUT_DIR / "stl_1950-2100_era5_cmip6_scenarios.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved plot to: {out_png}")
    plt.show()
