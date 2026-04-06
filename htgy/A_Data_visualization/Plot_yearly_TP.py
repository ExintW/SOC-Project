import sys
import os
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── PROJECT SETUP ────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR  # Path objects

# ─── OUTPUT FOLDER ────────────────────────────────────────────────────────────
plot_dir = OUTPUT_DIR / "Precipitation_Graphs"
plot_dir.mkdir(parents=True, exist_ok=True)

# ─── FUNCTIONS ────────────────────────────────────────────────────────────────
def compute_era5_tp_annual_stats(start_year, end_year):
    """
    Compute annual total precipitation statistics from ERA5 .nc files.

    For each year:
    1. Read monthly tp
    2. Convert to mm/month
    3. Sum 12 months at each point -> annual precipitation map
    4. Compute spatial mean and spatial std of the annual map
    """
    annual_records = []
    annual_maps = {}

    for year in range(start_year, end_year + 1):
        nc_file = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{year}.nc"

        with nc.Dataset(nc_file) as ds:
            tp = ds.variables["tp"][:]   # shape: (12, n_points)

            # ERA5 tp usually in meters water equivalent per day or per time step
            # Your original code uses *30*1000, so keep consistent with that
            tp_mm = tp * 30 * 1000.0     # mm/month

            # annual precipitation at each point
            annual_map = np.sum(tp_mm, axis=0)   # shape: (n_points,)

            annual_mean = np.nanmean(annual_map)
            annual_std = np.nanstd(annual_map, ddof=1)

            annual_records.append({
                "year": year,
                "annual_total_precipitation": float(annual_mean),
                "annual_spatial_std_precipitation": float(annual_std)
            })

            annual_maps[year] = annual_map

    annual_df = pd.DataFrame(annual_records)
    return annual_df, annual_maps


def get_era5_coordinates(sample_year=1950):
    """
    Read coordinate variables from one ERA5 resampled file.
    Tries several common coordinate names.
    """
    nc_file = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{sample_year}.nc"

    with nc.Dataset(nc_file) as ds:
        print("Available variables in ERA5 file:")
        print(list(ds.variables.keys()))

        lon_candidates = ["lon", "longitude", "LON", "LONGITUDE", "x", "X"]
        lat_candidates = ["lat", "latitude", "LAT", "LATITUDE", "y", "Y"]

        lon_name = next((name for name in lon_candidates if name in ds.variables), None)
        lat_name = next((name for name in lat_candidates if name in ds.variables), None)

        if lon_name is None or lat_name is None:
            raise KeyError(
                "Could not find coordinate variables. "
                f"Available variables are: {list(ds.variables.keys())}"
            )

        lon = ds.variables[lon_name][:]
        lat = ds.variables[lat_name][:]

    return lon, lat


def plot_precip_bar(annual_df, out_png):
    plt.figure(figsize=(18, 8))

    plt.bar(
        annual_df["year"],
        annual_df["annual_total_precipitation"],
        yerr=annual_df["annual_spatial_std_precipitation"],
        capsize=2
    )

    plt.xlabel("Year", fontsize=16)
    plt.ylabel("Annual total precipitation (mm)", fontsize=16)
    plt.title("Annual Total Precipitation with Spatial Standard Deviation (1950–2024)", fontsize=16)

    plt.xticks(np.arange(1950, 2025, 10), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(1949.5, 2024.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()


def plot_precip_map(lon, lat, annual_map, year, out_png):
    plt.figure(figsize=(12, 6))

    sc = plt.scatter(
        lon,
        lat,
        c=annual_map,
        s=8
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("Annual total precipitation (mm)", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.xlabel("Longitude", fontsize=16)
    plt.ylabel("Latitude", fontsize=16)
    plt.title(f"Precipitation Map, {year}", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.savefig(out_png, dpi=300)



# ─── MAIN SCRIPT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # =========================================================================
    # 1) Compute annual precipitation statistics for 1950–2024
    # =========================================================================
    annual_df, annual_maps = compute_era5_tp_annual_stats(1950, 2024)

    # save annual CSV
    annual_csv = plot_dir / "annual_total_precipitation_spatialstd_1950_2024.csv"
    annual_df.to_csv(annual_csv, index=False, encoding="utf-8-sig")
    print(f"Saved annual precipitation CSV to: {annual_csv}")

    # =========================================================================
    # 2) Plot annual bar chart with spatial std
    # =========================================================================
    bar_png = plot_dir / "annual_total_precipitation_with_spatialstd_1950_2024_bar.png"
    plot_precip_bar(annual_df, bar_png)
    print(f"Saved bar chart to: {bar_png}")

    # =========================================================================
    # 3) Get coordinates for mapping
    # =========================================================================
    lon, lat = get_era5_coordinates(sample_year=1950)

    # =========================================================================
    # 4) Plot decadal maps, 1950–2020
    # =========================================================================
    map_years = list(range(1950, 2021, 10))

    for year in map_years:
        if year not in annual_maps:
            print(f"Skipping map for {year}, data not found.")
            continue

        out_png = plot_dir / f"precipitation_map_{year}.png"
        plot_precip_map(lon, lat, annual_maps[year], year, out_png)
        print(f"Saved map: {out_png}")