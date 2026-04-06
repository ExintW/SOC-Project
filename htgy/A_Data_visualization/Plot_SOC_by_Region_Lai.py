#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annual mean and intra-annual standard deviation of Total_C for:
  1) High vegetation sedimentation area
  2) Low vegetation erosion area

For each year:
  - Compute annual mean LAI and Otsu threshold
  - Classify each grid as High_LAI / Low_LAI
  - For each month, compute group mean Total_C
  - For each year, compute:
      annual mean = mean of 12 monthly group means
      intra-annual std = std of 12 monthly group means
  - Save CSV
  - Plot annual mean with shaded ± intra-annual std
"""

# =============================================================================
# 1) Imports and paths
# =============================================================================
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR


# =============================================================================
# 2) Configuration
# =============================================================================
START_YEAR = 1950
END_YEAR   = 2024

LAI_FILES = {
    "past":    Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_1950-2000.nc",
    "present": Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2001-2014.nc",
    "future":  Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_245.nc",
}

PAST_SOC_DIR    = Path(OUTPUT_DIR) / "Data" / "SOC_Past 2"
PRESENT_SOC_DIR = Path(OUTPUT_DIR) / "Data" / "SOC_Present 7"

METRIC = "Total_C"

OUT_CSV = Path(OUTPUT_DIR) / "annual_Total_C_intra_annual_std_highveg_sedimentation_lowveg_erosion_1950_2024.csv"
OUT_FIG = Path(OUTPUT_DIR) / "annual_Total_C_intra_annual_std_highveg_sedimentation_lowveg_erosion_1950_2024.png"


# =============================================================================
# 3) Helper functions
# =============================================================================
def select_lai_file(year: int) -> Path:
    if year <= 2000:
        return LAI_FILES["past"]
    elif year <= 2014:
        return LAI_FILES["present"]
    else:
        return LAI_FILES["future"]


def soc_base_dir_for_year(year: int) -> Path:
    return PAST_SOC_DIR if year <= 2006 else PRESENT_SOC_DIR


def read_lai_group(year: int):
    """
    Read annual mean LAI for the year and classify each grid into High_LAI / Low_LAI.
    """
    lai_path = select_lai_file(year)
    ds_lai = xr.open_dataset(lai_path)

    lai_sel = ds_lai["lai"].sel(time=slice(f"{year}-01", f"{year}-12"))
    lai_mean = lai_sel.mean(dim="time")

    lats = ds_lai["lat"].values
    lons = ds_lai["lon"].values
    lai_vals = lai_mean.values
    ds_lai.close()

    valid = ~np.isnan(lai_vals)
    thr = threshold_otsu(lai_vals[valid])

    lai_df = pd.DataFrame({
        "LAT": lats,
        "LON": lons,
        "LAI": lai_vals,
    }).dropna(subset=["LAI"])

    lai_df["LAI_group"] = np.where(lai_df["LAI"] > thr, "High_LAI", "Low_LAI")
    lai_df["lat_r"] = lai_df["LAT"].round(4)
    lai_df["lon_r"] = lai_df["LON"].round(4)

    return lai_df, thr


def read_soc_month(year: int, month: int, metric: str = "Total_C") -> pd.DataFrame:
    """
    Read one monthly SOC parquet.
    """
    base_dir = soc_base_dir_for_year(year)
    fp = base_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"

    if not fp.exists():
        raise FileNotFoundError(f"Missing SOC file: {fp}")

    df = pd.read_parquet(fp)

    required_cols = ["Region", "LAT", "LON", metric]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in {fp}")

    df = df[required_cols].copy()
    df = df[df[metric].notna()]
    df["lat_r"] = df["LAT"].round(4)
    df["lon_r"] = df["LON"].round(4)

    return df


# =============================================================================
# 4) Main calculation
# =============================================================================
records = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"Processing {year} ...")

    lai_df, thr = read_lai_group(year)

    highveg_sed_monthly = []
    lowveg_ero_monthly = []

    for month in range(1, 13):
        soc_df = read_soc_month(year, month, metric=METRIC)

        merged = pd.merge(
            soc_df,
            lai_df[["lat_r", "lon_r", "LAI_group"]],
            on=["lat_r", "lon_r"],
            how="inner"
        )

        if merged.empty:
            highveg_sed_monthly.append(np.nan)
            lowveg_ero_monthly.append(np.nan)
            continue

        highveg_sed = merged[
            (merged["Region"] == "sedimentation area") &
            (merged["LAI_group"] == "High_LAI")
        ]

        lowveg_ero = merged[
            (merged["Region"] == "erosion area") &
            (merged["LAI_group"] == "Low_LAI")
        ]

        highveg_sed_monthly.append(highveg_sed[METRIC].mean() if not highveg_sed.empty else np.nan)
        lowveg_ero_monthly.append(lowveg_ero[METRIC].mean() if not lowveg_ero.empty else np.nan)

    highveg_sed_monthly = np.array(highveg_sed_monthly, dtype=float)
    lowveg_ero_monthly = np.array(lowveg_ero_monthly, dtype=float)

    rec = {
        "year": year,
        "lai_threshold": thr,

        "highveg_sedimentation_mean": np.nanmean(highveg_sed_monthly),
        "highveg_sedimentation_intra_annual_std": np.nanstd(highveg_sed_monthly, ddof=1),

        "lowveg_erosion_mean": np.nanmean(lowveg_ero_monthly),
        "lowveg_erosion_intra_annual_std": np.nanstd(lowveg_ero_monthly, ddof=1),
    }

    # optional: save monthly values too
    for m in range(12):
        rec[f"highveg_sedimentation_month_{m+1:02d}"] = highveg_sed_monthly[m]
        rec[f"lowveg_erosion_month_{m+1:02d}"] = lowveg_ero_monthly[m]

    records.append(rec)

summary = pd.DataFrame(records).sort_values("year")
summary.to_csv(OUT_CSV, index=False)

print(f"\nSaved CSV to:\n{OUT_CSV}")


# =============================================================================
# 5) Plot
# =============================================================================
plt.figure(figsize=(16, 7))

x = summary["year"]

y1 = summary["highveg_sedimentation_mean"]
s1 = summary["highveg_sedimentation_intra_annual_std"]

plt.plot(
    x, y1,
    linewidth=2.5,
    label="High vegetation sedimentation area"
)
plt.fill_between(
    x,
    y1 - s1,
    y1 + s1,
    alpha=0.25
)

y2 = summary["lowveg_erosion_mean"]
s2 = summary["lowveg_erosion_intra_annual_std"]

plt.plot(
    x, y2,
    linewidth=2.5,
    label="Low vegetation erosion area"
)
plt.fill_between(
    x,
    y2 - s2,
    y2 + s2,
    alpha=0.25
)

plt.xlabel("Year", fontsize=16)
plt.ylabel("Annual mean Total_C", fontsize=16)
plt.title("Annual mean Total_C with intra-annual standard deviation", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc="upper left", fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUT_FIG, dpi=600, bbox_inches="tight")
plt.show()

print(f"Saved figure to:\n{OUT_FIG}")