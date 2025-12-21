#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count how many sedimentation-area grid cells exist in each month.

For a given dataset (base_dir) and year range:
  - Reads SOC_terms_{year}_{month}_River.parquet
  - Filters rows where Region == "sedimentation area"
  - Counts unique grid cells (LAT, LON) in that month
  - Also records total number of grid cells and fraction sedimentation

Outputs:
  OUTPUT_DIR / "sedimentation_grid_counts_<label>.csv"
"""

import os
import sys
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1) Make OUTPUT_DIR available
# ──────────────────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import OUTPUT_DIR

# ──────────────────────────────────────────────────────────────────────────────
# 2) Configuration
# ──────────────────────────────────────────────────────────────────────────────
# Choose which dataset you want to analyze:

# Example 1: PAST (1950–2006)
# BASE_DIR   = OUTPUT_DIR / "Data" / "SOC_Past 2"
# START_YEAR = 1950
# END_YEAR   = 2006
# LABEL      = "past_1950_2006"

# Example 2: PRESENT (2007–2024)
# BASE_DIR   = OUTPUT_DIR / "Data" / "SOC_Present 7"
# START_YEAR = 2007
# END_YEAR   = 2024
# LABEL      = "present_2007_2024"

# Example 3: FUTURE scenario 585 (2025–2100)
# BASE_DIR   = OUTPUT_DIR / "Data" / "SOC_Future 7" / "585"
# START_YEAR = 2025
# END_YEAR   = 2100
# LABEL      = "future_585_2025_2100"

# 👉 Pick one of the above or define your own; for now I set PRESENT as default:
BASE_DIR   = OUTPUT_DIR / "Data" / "SOC_Present 7"
START_YEAR = 2024
END_YEAR   = 2024
LABEL      = "present__2024"


# ──────────────────────────────────────────────────────────────────────────────
# 3) Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    records = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            fn = f"SOC_terms_{year}_{month:02d}_River.parquet"
            fp = BASE_DIR / fn

            if not fp.exists():
                # If the file does not exist, skip this month
                print(f"Missing file, skipping: {fp}")
                continue

            df = pd.read_parquet(fp)

            # Optional: filter out rows without Total_C, if you want only valid cells
            if "Total_C" in df.columns:
                df = df[df["Total_C"].notna()]

            # Total number of unique grid cells (LAT, LON)
            total_cells = (
                df[["LAT", "LON"]]
                .drop_duplicates()
                .shape[0]
            )

            # Sedimentation-area cells
            if "Region" not in df.columns:
                raise KeyError(f"'Region' column not found in {fp}")

            df_sed = df[df["Region"] == "sedimentation area"]
            sed_cells = (
                df_sed[["LAT", "LON"]]
                .drop_duplicates()
                .shape[0]
            )

            frac_sed = sed_cells / total_cells if total_cells > 0 else 0.0

            records.append({
                "year":        year,
                "month":       month,
                "n_total_cells":       total_cells,
                "n_sedimentation_cells": sed_cells,
                "frac_sedimentation":   frac_sed,
            })

    if not records:
        raise RuntimeError("No records created. Check BASE_DIR and year range.")

    out_df = pd.DataFrame(records).sort_values(["year", "month"])

    out_fp = OUTPUT_DIR / f"sedimentation_grid_counts_{LABEL}.csv"
    out_df.to_csv(out_fp, index=False)

    print("\nAll done — sedimentation grid counts saved to:")
    print(f"  {out_fp}\n")


if __name__ == "__main__":
    main()
