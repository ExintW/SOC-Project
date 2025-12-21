#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annual SOC-term summary (1950–2024) by LAI group (High_LAI / Low_LAI).

For each year:
  1) Use CMIP6 LAI to compute annual-mean LAI and an Otsu threshold,
     classify grid points as High_LAI / Low_LAI.
  2) Read 12 monthly SOC Parquet files for that year:
       - 1950–2006: from "SOC_Past 2"
       - 2007–2024: from "SOC_Present 7"
     and compute annual means of all numeric SOC variables per grid cell.
  3) Merge SOC with LAI groups by rounded coordinates.
  4) For each LAI group, compute the spatial mean of selected metrics:
       C_fast, C_slow, Total_C,
       Erosion_fast, Erosion_slow,
       Deposition_fast, Deposition_slow,
       Vegetation_fast, Vegetation_slow,
       Reaction_fast, Reaction_slow,
       Lost_SOC_River, Trapped_SOC_Dam,
       E_t_ha_month  (erosion modulus, t/ha/month)
  5) Output a wide CSV where:
       - each row = a year
       - columns = metric_group, e.g. C_fast_High_LAI, C_fast_Low_LAI, ...

Output:
  OUTPUT_DIR / "annual_soc_terms_by_LAI_1950_2024.csv"
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from skimage.filters import threshold_otsu  # pip install scikit-image

# ─── make globals available ──────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR

# ─── configuration ───────────────────────────────────────────────────────────
START_YEAR = 1950
END_YEAR   = 2024

# LAI NetCDFs (same as your existing LAI script)
LAI_FILES = {
    "past":    Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_1950-2000.nc",
    "present": Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2001-2014.nc",
    "future":  Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_245.nc",
}

# SOC Parquet dirs
PAST_SOC_DIR    = Path(OUTPUT_DIR) / "Data" / "SOC_Past 2"
PRESENT_SOC_DIR = Path(OUTPUT_DIR) / "Data" / "SOC_Present 7"

# Metrics we want to mirror from your original annual summary, plus E_t_ha_month
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
    "Trapped_SOC_Dam",
    "E_t_ha_month",  # erosion modulus (t/ha/month)
]


def select_lai_file(year: int) -> Path:
    """Return the appropriate LAI NetCDF path for a given year."""
    if year <= 2000:
        return LAI_FILES["past"]
    elif year <= 2014:
        return LAI_FILES["present"]
    else:
        return LAI_FILES["future"]


def soc_base_dir_for_year(year: int) -> Path:
    """Return the SOC Parquet directory for a given year."""
    return PAST_SOC_DIR if year <= 2006 else PRESENT_SOC_DIR


def read_soc_annual_means(year: int) -> pd.DataFrame:
    """
    Read 12 monthly SOC Parquet files for a given year and compute
    annual means of all numeric columns per grid cell (LAT, LON).

    Returns DataFrame with columns:
      LAT, LON, <all numeric SOC variables as annual means>
    """
    base_dir = soc_base_dir_for_year(year)

    monthly = []
    for m in range(1, 13):
        fn = f"SOC_terms_{year}_{m:02d}_River.parquet"
        fp = base_dir / fn
        if not fp.exists():
            raise FileNotFoundError(f"Missing SOC file: {fp}")

        df = pd.read_parquet(fp)
        # 保留所有数值列，先过滤掉 Total_C 缺失的行
        if "Total_C" in df.columns:
            df = df[df["Total_C"].notna()]
        monthly.append(df)

    all_df = pd.concat(monthly, ignore_index=True)

    # group by point, take mean of all numeric columns (annual mean)
    grouped = (
        all_df
        .groupby(["LAT", "LON"], as_index=False)
        .mean(numeric_only=True)
    )
    return grouped


def main():
    records = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Processing year {year} ...")

        # ── 1) LAI: annual mean & Otsu threshold → High/Low groups ──────────
        lai_path = select_lai_file(year)
        ds_lai   = xr.open_dataset(lai_path)

        lai_sel  = ds_lai["lai"].sel(time=slice(f"{year}-01", f"{year}-12"))
        lai_mean = lai_sel.mean(dim="time")

        lats     = ds_lai["lat"].values
        lons     = ds_lai["lon"].values
        lai_vals = lai_mean.values
        ds_lai.close()

        # Compute Otsu threshold on valid LAI values
        valid = ~np.isnan(lai_vals)
        thr   = threshold_otsu(lai_vals[valid])

        lai_df = pd.DataFrame({
            "LAT": lats,
            "LON": lons,
            "LAI": lai_vals,
        }).dropna(subset=["LAI"])

        lai_df["group"] = np.where(lai_df["LAI"] > thr, "High_LAI", "Low_LAI")

        # ── 2) SOC annual means at each point ────────────────────────────────
        soc_annual = read_soc_annual_means(year)

        # ── 3) merge LAI groups with SOC by rounded coordinates ──────────────
        for df_ in (lai_df, soc_annual):
            df_["lat_r"] = df_["LAT"].round(4)
            df_["lon_r"] = df_["LON"].round(4)

        merged = pd.merge(
            lai_df[["lat_r", "lon_r", "group"]],
            soc_annual,
            on=["lat_r", "lon_r"],
            how="inner",
        )

        if merged.empty:
            print(f"  Warning: merged LAI + SOC is empty for year {year}")
            continue

        # ── 4) group means for selected metrics ──────────────────────────────
        # We only keep metrics that actually exist in the data
        available_metrics = [
            m for m in METRICS_TO_KEEP
            if m in merged.columns
        ]

        # groupby("group") → mean of metrics across points
        group_means = (
            merged
            .groupby("group")[available_metrics]
            .mean(numeric_only=True)
        )

        # Build one record per year (wide: metrics × groups → columns)
        rec = {"year": year, "lai_threshold": thr}

        for grp in ["High_LAI", "Low_LAI"]:
            if grp not in group_means.index:
                # In case a group is missing (very unlikely), fill NaNs
                for m in available_metrics:
                    rec[f"{m}_{grp}"] = np.nan
                continue

            for m in available_metrics:
                rec[f"{m}_{grp}"] = group_means.loc[grp, m]

        records.append(rec)

    # ── 5) Build final DataFrame (one row per year) ──────────────────────────
    if not records:
        raise RuntimeError("No records created. Check data paths and years.")

    summary = pd.DataFrame(records).sort_values("year")

    # Save to CSV
    out_fp = Path(OUTPUT_DIR) / "annual_soc_terms_by_LAI_1950_2024.csv"
    summary.to_csv(out_fp, index=False)

    print("\nAll done — annual SOC-term summary by LAI group saved to:")
    print(f"  {out_fp}\n")


if __name__ == "__main__":
    main()
