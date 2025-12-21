#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annual SOC-term summary (1950–2024) by Region
  (erosion area vs sedimentation area).

For each year:
  1) Read 12 monthly SOC Parquet files:
       - 1950–2006:  Data/SOC_Past 2
       - 2007–2024:  Data/SOC_Present 7
  2) Filter out rows lacking Total_C.
  3) For each grid cell (LAT, LON, Region), compute the annual mean of all
     numeric SOC variables (including E_t_ha_month).
  4) For each Region ("erosion area", "sedimentation area"), compute the
     spatial mean of selected metrics:
        C_fast, C_slow, Total_C,
        Erosion_fast, Erosion_slow,
        Deposition_fast, Deposition_slow,
        Vegetation_fast, Vegetation_slow,
        Reaction_fast, Reaction_slow,
        Lost_SOC_River, Trapped_SOC_Dam,
        E_t_ha_month
  5) Output a wide CSV where:
        - each row = a year
        - columns = metric_erosion_area, metric_sedimentation_area

Output:
  OUTPUT_DIR / "annual_soc_terms_by_region_1950_2024.csv"
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── make globals available ──────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import OUTPUT_DIR  # root Output folder

# ─── configuration ───────────────────────────────────────────────────────────
START_YEAR = 1950
END_YEAR   = 2024

PAST_SOC_DIR    = Path(OUTPUT_DIR) / "Data" / "SOC_Past 2"
PRESENT_SOC_DIR = Path(OUTPUT_DIR) / "Data" / "SOC_Present 7"

# Metrics to mirror from your original annual summary, plus E_t_ha_month
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

# Map Region string → suffix for column names
REGION_SUFFIX = {
    "erosion area": "erosion_area",
    "sedimentation area": "sedimentation_area",
}


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
        if "Total_C" in df.columns:
            df = df[df["Total_C"].notna()]
        if "Region" not in df.columns:
            raise KeyError(f"'Region' column not found in {fp}")

        monthly.append(df)

    all_df = pd.concat(monthly, ignore_index=True)

    # Group by Region + point and average all numeric columns over 12 months
    grouped = (
        all_df
        .groupby(["Region", "LAT", "LON"], as_index=False)
        .mean(numeric_only=True)
    )
    return grouped


def main():
    records = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Processing year {year} ...")

        soc_annual = read_soc_annual_means_with_region(year)

        # available metrics = ones that exist in the data
        available_metrics = [
            m for m in METRICS_TO_KEEP
            if m in soc_annual.columns
        ]

        if not available_metrics:
            print(f"  Warning: no requested metrics found for year {year}")
            continue

        # For each Region, compute spatial mean of those metrics
        region_means = (
            soc_annual
            .groupby("Region")[available_metrics]
            .mean(numeric_only=True)
        )

        rec = {"year": year}

        for region_name, suffix in REGION_SUFFIX.items():
            if region_name not in region_means.index:
                # If this Region is missing for that year, fill with NaN
                for m in available_metrics:
                    rec[f"{m}_{suffix}"] = np.nan
                continue

            for m in available_metrics:
                rec[f"{m}_{suffix}"] = region_means.loc[region_name, m]

        records.append(rec)

    if not records:
        raise RuntimeError("No records created. Check data paths and years.")

    summary = pd.DataFrame(records).sort_values("year")

    out_fp = Path(OUTPUT_DIR) / "annual_soc_terms_by_region_1950_2024.csv"
    summary.to_csv(out_fp, index=False)

    print("\nAll done — annual SOC-term summary by Region saved to:")
    print(f"  {out_fp}\n")


if __name__ == "__main__":
    main()
