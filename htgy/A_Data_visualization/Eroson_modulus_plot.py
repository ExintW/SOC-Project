import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# =============================================================================
# 1) Configuration & File-path Setup
# =============================================================================
past_dir = OUTPUT_DIR / "Data" / "SOC_Past 2"
present_dir = OUTPUT_DIR / "Data" / "SOC_Present 7"

plot_dir = OUTPUT_DIR / "Erosion_Modulus_Graphs"
plot_dir.mkdir(parents=True, exist_ok=True)

map_years = list(range(1950, 2025, 10))   # 1950, 1960, ..., 2020

# =============================================================================
# 2) Helper Function to Read One Year of Monthly Parquet Files
# =============================================================================
def read_one_year_monthly_data(year, year_dir):
    monthly_frames = []

    for month in range(1, 13):
        path = year_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"

        if not path.exists():
            print(f"Missing file: {path}")
            continue

        df = pd.read_parquet(path)

        required_cols = ["LAT", "LON", "E_t_ha_month"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"{missing_cols} not found in {path.name}")

        df_sub = df[["LAT", "LON", "E_t_ha_month"]].copy()
        df_sub["month"] = month
        monthly_frames.append(df_sub)

    if len(monthly_frames) == 0:
        return None

    df_year = pd.concat(monthly_frames, ignore_index=True)
    return df_year

# =============================================================================
# 3) Compute Annual Total Erosion Modulus and Annual Spatial Std, 1950–2024
# =============================================================================
annual_records = []
monthly_records = []

for year in range(1950, 2025):
    year_dir = past_dir if year <= 2006 else present_dir
    df_year = read_one_year_monthly_data(year, year_dir)

    if df_year is None:
        print(f"No valid files found for year {year}")
        continue

    # monthly whole-plateau summary for reference
    monthly_summary = (
        df_year.groupby("month", as_index=False)["E_t_ha_month"]
        .agg(
            monthly_mean_erosion_modulus="mean",
            monthly_spatial_std_erosion_modulus="std"
        )
    )
    monthly_summary["year"] = year
    monthly_records.append(monthly_summary)

    # annual map = mean of 12 monthly values at each point
    df_map = (
        df_year.groupby(["LAT", "LON"], as_index=False)["E_t_ha_month"]
        .mean()
        .rename(columns={"E_t_ha_month": "annual_mean_point_erosion"})
    )

    # annual total erosion modulus = annual monthly-average map mean × 12
    annual_total = df_map["annual_mean_point_erosion"].mean() * 12

    # spatial std of the annual map
    annual_spatial_std = df_map["annual_mean_point_erosion"].std()

    annual_records.append({
        "year": year,
        "annual_total_erosion_modulus": float(annual_total),
        "annual_spatial_std_erosion_modulus": float(annual_spatial_std)
    })

# assemble annual dataframe
annual_stats = pd.DataFrame(annual_records).sort_values("year")

# assemble monthly dataframe
monthly_df = pd.concat(monthly_records, ignore_index=True)
monthly_df = monthly_df[[
    "year",
    "month",
    "monthly_mean_erosion_modulus",
    "monthly_spatial_std_erosion_modulus"
]]

# =============================================================================
# 4) Save CSV Files
# =============================================================================
monthly_csv = plot_dir / "monthly_erosion_modulus_mean_spatialstd_1950_2024.csv"
monthly_df.to_csv(monthly_csv, index=False, encoding="utf-8-sig")
print(f"Saved monthly CSV to: {monthly_csv}")

annual_csv = plot_dir / "annual_total_erosion_modulus_spatialstd_1950_2024.csv"
annual_stats.to_csv(annual_csv, index=False, encoding="utf-8-sig")
print(f"Saved annual CSV to: {annual_csv}")

# =============================================================================
# 5) Plot Annual Bar Chart with Spatial Std Error Bars
# =============================================================================
plt.figure(figsize=(18, 8))

plt.bar(
    annual_stats["year"],
    annual_stats["annual_total_erosion_modulus"],
    yerr=annual_stats["annual_spatial_std_erosion_modulus"],
    capsize=2
)

plt.xlabel("Year", fontsize=14)
plt.ylabel("Annual total erosion modulus", fontsize=14)
plt.title("Annual Total Erosion Modulus with Spatial Standard Deviation (1950–2024)", fontsize=16)

plt.xticks(np.arange(1950, 2025, 10), fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(1949.5, 2024.5)

plt.tight_layout()

bar_png = plot_dir / "annual_total_erosion_modulus_with_spatialstd_1950_2024_bar.png"
plt.savefig(bar_png, dpi=300)
print(f"Saved bar chart to: {bar_png}")

plt.show()

# =============================================================================
# 6) Plot Decadal Maps, 1950–2020
# =============================================================================
for year in map_years:
    year_dir = past_dir if year <= 2006 else present_dir
    df_year = read_one_year_monthly_data(year, year_dir)

    if df_year is None:
        print(f"No valid files found for map year {year}")
        continue

    # annual map = average of 12 months at each point
    df_map = (
        df_year.groupby(["LAT", "LON"], as_index=False)["E_t_ha_month"]
        .mean()
        .rename(columns={"E_t_ha_month": "annual_mean_erosion_modulus"})
    )

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        df_map["LON"],
        df_map["LAT"],
        c=df_map["annual_mean_erosion_modulus"],
        s=8
    )
    plt.colorbar(sc, label="Annual mean erosion modulus")
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.title(f"Erosion Modulus Map, {year}", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    out_map = plot_dir / f"erosion_modulus_map_{year}.png"
    plt.savefig(out_map, dpi=300)
    print(f"Saved map: {out_map}")

    plt.show()