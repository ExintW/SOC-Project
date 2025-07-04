import sys
import os
import netCDF4 as nc
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

USE_PARQUET = True

def compute_era5_tp(start_year, end_year):
    """Compute annual mean tp (mm/year) from ERA5 .nc files for 1950–2024."""
    tp_hist = []
    years_hist = list(range(start_year, end_year+1))
    for year in years_hist:
        nc_file = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{year}.nc"
        with nc.Dataset(nc_file) as ds:
            # tp in meters per second, multiply by 30 days and by 1000 to get mm
            tp = ds.variables['tp'][:]     # shape (12, n_points)
            tp_mm = tp * 30 * 1000.0       # mm/month
            # sum over months → annual total per point, then average spatially
            annual = np.sum(tp_mm, axis=0) # (n_points,)
            tp_hist.append(np.mean(annual))
    return years_hist, tp_hist


def compute_cmip6_tp(scenario_tag, year_start=2025, year_end=2100):
    """
    Compute annual mean tp (mm/year) for a single CMIP6 scenario,
    reading pr (kg m^-2 s^-1) from 2015–2100 file and converting to mm/month.
    """
    fn = f"resampled_pr_points_2015-2100_{scenario_tag}.nc"
    ds = nc.Dataset(PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / fn)
    pr = ds.variables['pr'][:]           # shape (time, n_points)
    # convert to mm/month
    pr_mm = pr * 86400.0 * 30.0

    years = list(range(year_start, year_end+1))
    tp_scenario = []

    for yr in years:
        # compute the index of the first month of this year in the 2015–2100 series
        idx0 = (yr - 2015) * 12
        idx1 = idx0 + 12
        annual = np.sum(pr_mm[idx0:idx1, :], axis=0)  # sum each point over 12 months
        tp_scenario.append(np.mean(annual))
    ds.close()
    return years, tp_scenario


if __name__ == "__main__":
    # 1) historical ERA5
    hist_years, hist_tp = compute_era5_tp(1950, 2024)
    last_year, last_tp = hist_years[-1], hist_tp[-1]

    # 2) CMIP6 scenarios
    scenarios = {
        "126": "SSP1-2.6",
        "245": "SSP2-4.5",
        "370": "SSP3-7.0",
        "585": "SSP5-8.5",
    }
    cmip_data = {}
    for tag, name in scenarios.items():
        yrs, tp_vals = compute_cmip6_tp(tag, 2025, 2100)
        # prepend the 2024 historical point for continuity into 2025
        yrs    = [last_year] + yrs
        tp_vals= [last_tp]   + tp_vals
        cmip_data[name] = (yrs, tp_vals)

    # 3) Plot
    plt.figure(figsize=(10, 6))

    # plot each scenario (now continuous across 2024→2025)
    for name, (yrs, tp_vals) in cmip_data.items():
        plt.plot(yrs, tp_vals, linewidth=1, label=name)

    # overlay the thick black ERA5 curve
    plt.plot(hist_years, hist_tp, color='black', linewidth=1, label='1950–2024 (ERA5)')

    plt.xlabel('Year')
    plt.ylabel('Annual Total Precipitation (mm)')
    plt.title('Precipitation 1950–2100: Historical + CMIP6 Scenarios')
    plt.legend()
    plt.tight_layout()

    out_png = OUTPUT_DIR / "tp_1950-2100_era5_cmip6_scenarios.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved figure to: {out_png}")
    plt.show()
