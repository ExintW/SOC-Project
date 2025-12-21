#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annual overall SOC-term summary (1950–2024), no spatial splitting.

For each year:
  1) Read 12 monthly SOC Parquet files:
       - 1950–2006:  Data/SOC_Past 2
       - 2007–2024:  Data/SOC_Present 7
  2) Filter out rows lacking Total_C.
  3) For each grid cell (LAT, LON), compute the annual mean of all numeric
     SOC variables.
  4) Over all grid cells, compute the spatial mean of selected metrics:
       C_fast, C_slow, Total_C,
       Erosion_fast, Erosion_slow,
       Deposition_fast, Deposition_slow,
       Vegetation_fast, Vegetation_slow,
       Reaction_fast, Reaction_slow,
       Lost_SOC_River, Trapped_SOC_Dam,
       E_t_ha_month (erosion modulus, t/ha/month).
  5) Output a CSV where each row is a year and each column is one metric.

Output:
  OUTPUT_DIR / "annual_soc_terms_overall_1950_2024.csv"
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

# Metrics to summarize (same as region/LAI scripts + E_t_ha_month)
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


def soc_base_dir_for_year(year: int) -> Path:
    """Return the SOC Parquet base directory for a given year."""
    return PAST_SOC_DIR if year <= 2006 else PRESENT_SOC_DIR


def read_soc_annual_means(year: int) -> pd.DataFrame:
    """
    Read 12 monthly SOC Parquet files for a given year and compute
    annual means of all numeric columns per grid cell (LAT, LON).

    Returns:
      DataFrame with columns:
        LAT, LON, <annual mean of all numeric SOC variables>
    """
    base_dir = soc_base_dir_for_year(year)
    monthly = []

    for m in range(1, 13):
        fn = f"SOC_terms_{year}_{m:02d}_River.parquet"
        fp = base_dir / fn
        if not fp.exists():
            raise FileNotFoundError(f"Missing SOC file: {fp}")

        df = pd.read_parquet(fp)

        # Filter out rows without Total_C
        if "Total_C" in df.columns:
            df = df[df["Total_C"].notna()]

        monthly.append(df)

    all_df = pd.concat(monthly, ignore_index=True)

    # group by grid cell, take annual mean of all numeric columns
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

        soc_annual = read_soc_annual_means(year)

        # Only keep metrics that actually exist in the data
        available_metrics = [m for m in METRICS_TO_KEEP if m in soc_annual.columns]
        if not available_metrics:
            print(f"  Warning: no requested metrics found for year {year}")
            continue

        # Spatial mean over all grid cells
        means = soc_annual[available_metrics].mean(numeric_only=True)

        rec = {"year": year}
        for m in available_metrics:
            rec[m] = means[m]

        records.append(rec)

    if not records:
        raise RuntimeError("No records created. Check data paths and years.")

    summary = pd.DataFrame(records).sort_values("year")

    out_fp = Path(OUTPUT_DIR) / "annual_soc_terms_overall_1950_2024.csv"
    summary.to_csv(out_fp, index=False)

    print("\nAll done — overall annual SOC-term summary saved to:")
    print(f"  {out_fp}\n")


if __name__ == "__main__":
    main()
