#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annual SOC-term summary (1950–2024) by Region × LAI group.

For each year:
  1) Use CMIP6 LAI to compute annual-mean LAI and an Otsu threshold,
     classify grid points as High_LAI / Low_LAI.
  2) Read 12 monthly SOC Parquet files:
       - 1950–2006:  Data/SOC_Past 2
       - 2007–2024:  Data/SOC_Present 7
     and compute annual means per (Region, LAT, LON) of all numeric SOC
     variables (including E_t_ha_month).
  3) Merge SOC with LAI using rounded coordinates.
  4) For each combination of Region × LAI group, compute the spatial mean
     of selected metrics:
        C_fast, C_slow, Total_C,
        Erosion_fast, Erosion_slow,
        Deposition_fast, Deposition_slow,
        Vegetation_fast, Vegetation_slow,
        Reaction_fast, Reaction_slow,
        Lost_SOC_River,
        E_t_ha_month
  5) Output a wide CSV where:
        - each row = a year
        - columns = metric_region_LAI, e.g.
          C_fast_erosion_HighLAI,
          C_fast_sedimentation_LowLAI,
          E_t_ha_month_erosion_HighLAI, etc.

Output:
  OUTPUT_DIR / "annual_soc_terms_by_region_LAI_1950_2024.csv"
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from skimage.filters import threshold_otsu  # pip install scikit-image

# ──────────────────────────────────────────────────────────────────────────────
# 0) Make globals available
# ──────────────────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR

# ──────────────────────────────────────────────────────────────────────────────
# 1) Configuration
# ──────────────────────────────────────────────────────────────────────────────
START_YEAR = 1950
END_YEAR   = 2024

# LAI NetCDF files (same structure as your LAI script)
LAI_FILES = {
    "past":    Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_1950-2000.nc",
    "present": Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2001-2014.nc",
    "future":  Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_245.nc",
}

# SOC Parquet dirs
PAST_SOC_DIR    = Path(OUTPUT_DIR) / "Data" / "SOC_Past 2"
PRESENT_SOC_DIR = Path(OUTPUT_DIR) / "Data" / "SOC_Present 7"

# Metrics to summarize, plus erosion modulus
METRICS_TO_KEEP = [
    "C_fast",
    "C_slow",
    "Total_C",
    "Erosion_fast",
    "Erosion_slow",
    "Deposition_fast",
    "Deposition_slow",
    "Vegetation_fast",
    "Vegetation_slow",
    "Reaction_fast",
    "Reaction_slow",
    "Lost_SOC_River",
    "E_t_ha_month",   # erosion modulus (t/ha/month)
]

# Region and LAI suffixes for column names
REGION_SUFFIX = {
    "erosion area":        "erosion",
    "sedimentation area":  "sedimentation",
}

LAI_SUFFIX = {
    "High_LAI": "HighLAI",
    "Low_LAI":  "LowLAI",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2) Helpers
# ──────────────────────────────────────────────────────────────────────────────
def select_lai_file(year: int) -> Path:
    """Return the appropriate LAI NetCDF path for a given year."""
    if year <= 2000:
        return LAI_FILES["past"]
    elif year <= 2014:
        return LAI_FILES["present"]
    else:
        return LAI_FILES["future"]


def soc_base_dir_for_year(year: int) -> Path:
    """Return the SOC Parquet base directory for a given year."""
    return PAST_SOC_DIR if year <= 2006 else PRESENT_SOC_DIR


def read_soc_annual_means_with_region(year: int) -> pd.DataFrame:
    """
    Read 12 monthly SOC Parquet files for a given year and compute
    annual means per (Region, LAT, LON).

    Returns DataFrame with columns:
      Region, LAT, LON, <annual mean of all numeric SOC columns>
    """
    base_dir = soc_base_dir_for_year(year)
    monthly = []

    for m in range(1, 13):
        fn = f"SOC_terms_{year}_{m:02d}_River.parquet"
        fp = base_dir / fn
        if not fp.exists():
            raise FileNotFoundError(f"Missing SOC file: {fp}")

        df = pd.read_parquet(fp)

        # Ensure Region exists and filter out rows without Total_C
        if "Region" not in df.columns:
            raise KeyError(f"'Region' column not found in {fp}")

        if "Total_C" in df.columns:
            df = df[df["Total_C"].notna()]

        monthly.append(df)

    all_df = pd.concat(monthly, ignore_index=True)

    # Group by Region + point and average all numeric columns over 12 months
    grouped = (
        all_df
        .groupby(["Region", "LAT", "LON"], as_index=False)
        .mean(numeric_only=True)
    )
    return grouped


# ──────────────────────────────────────────────────────────────────────────────
# 3) Main routine
# ──────────────────────────────────────────────────────────────────────────────
def main():
    records = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Processing year {year} ...")

        # 3.1) LAI: annual mean and Otsu threshold → High / Low
        lai_path = select_lai_file(year)
        ds_lai   = xr.open_dataset(lai_path)

        lai_sel  = ds_lai["lai"].sel(time=slice(f"{year}-01", f"{year}-12"))
        lai_mean = lai_sel.mean(dim="time")

        lats     = ds_lai["lat"].values
        lons     = ds_lai["lon"].values
        lai_vals = lai_mean.values
        ds_lai.close()

        valid = ~np.isnan(lai_vals)
        thr   = threshold_otsu(lai_vals[valid])

        # here we assume lats, lons, lai_vals are 1D and aligned by index
        lai_df = pd.DataFrame({
            "LAT": lats,
            "LON": lons,
            "LAI": lai_vals,
        }).dropna(subset=["LAI"])

        lai_df["LAI_group"] = np.where(lai_df["LAI"] > thr, "High_LAI", "Low_LAI")

        # 3.2) SOC annual means with Region
        soc_annual = read_soc_annual_means_with_region(year)

        # 3.3) Merge by rounded coordinates (Region stays in soc_annual)
        for df_ in (lai_df, soc_annual):
            df_["lat_r"] = df_["LAT"].round(4)
            df_["lon_r"] = df_["LON"].round(4)

        merged = pd.merge(
            soc_annual,
            lai_df[["lat_r", "lon_r", "LAI_group"]],
            on=["lat_r", "lon_r"],
            how="inner",
        )

        if merged.empty:
            print(f"  Warning: merged Region × LAI is empty for year {year}")
            continue

        # 3.4) Compute spatial means for each Region × LAI_group
        available_metrics = [m for m in METRICS_TO_KEEP if m in merged.columns]

        if not available_metrics:
            print(f"  Warning: no requested metrics found for year {year}")
            continue

        grouped_means = (
            merged
            .groupby(["Region", "LAI_group"])[available_metrics]
            .mean(numeric_only=True)
        )

        # 3.5) Build one record per year in wide format
        rec = {
            "year":          year,
            "lai_threshold": thr,
        }

        # target combinations
        for region_name, region_suffix in REGION_SUFFIX.items():
            for lai_group, lai_suffix in LAI_SUFFIX.items():
                col_suffix = f"{region_suffix}_{lai_suffix}"  # e.g. erosion_HighLAI
                if (region_name, lai_group) not in grouped_means.index:
                    # no cells for this combo in this year
                    for m in available_metrics:
                        rec[f"{m}_{col_suffix}"] = np.nan
                    continue

                row = grouped_means.loc[(region_name, lai_group)]
                for m in available_metrics:
                    rec[f"{m}_{col_suffix}"] = row[m]

        records.append(rec)

    # 3.6) Build final DataFrame and save
    if not records:
        raise RuntimeError("No records created. Check data paths and years.")

    summary = pd.DataFrame(records).sort_values("year")

    out_fp = Path(OUTPUT_DIR) / "annual_soc_terms_by_region_LAI_1950_2024.csv"
    summary.to_csv(out_fp, index=False)

    print("\nAll done — annual SOC terms by Region × LAI group saved to:")
    print(f"  {out_fp}\n")


if __name__ == "__main__":
    main()
