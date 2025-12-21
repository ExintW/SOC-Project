#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annual SOC-term summary for future scenarios (2025–2100).

Scenarios:
    126, 245, 370, 585

For each scenario and year:
  1) Read 12 monthly SOC Parquet files from:
       OUTPUT_DIR / "Data" / "SOC_Future 7" / <scenario>
  2) Filter out rows with missing Total_C.
  3) Compute annual mean of all numeric columns across all cells & months.
  4) Derive:
       - Trapped_SOC_Dam             (average concentration in sedimentation area)
       - Erosion_pct                 ( (Erosion_fast + Erosion_slow) / Total_C * 100 )
       - Deposition_pct              ( (Deposition_fast + Deposition_slow) / Total_C * 100 )
       - Vegetation_pct              ( (Vegetation_fast + Vegetation_slow) / Total_C * 100 )
       - Reaction_pct                ( (Reaction_fast + Reaction_slow) / Total_C * 100 )
       - River_lost_pct              ( Lost_SOC_River / Total_C * 100 )
       - SOC_trapped_dam_pct         (avg % of Total_C in Region == "sedimentation area")
       - SOC_trapped_low_point_pct   (avg % of Total_C in Low point == True)
  5) Adjust the 5 process % metrics to use previous-year Total_C as denominator
     (same logic as original annual_summary; for the first year, use its own Total_C).
  6) Save one CSV per scenario:
       OUTPUT_DIR / f"annual_future_summary_{scenario}_2025_2100.csv"

Each CSV:
    - one row per year
    - columns = all mean variables (after dropping some RUSLE helper columns)
      plus the 7 percentage metrics.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1) Globals & configuration
# ──────────────────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import OUTPUT_DIR  # root Output folder

SCENARIOS  = ["126", "245", "370", "585"]
START_YEAR = 2025
END_YEAR   = 2100

# Columns we do not want in the final "means" output
TO_DROP_MEANS = [
    "LAT",
    "LON",
    "C_factor_month",
    "K_factor_month",
    "LS_factor_month",
    "P_factor_month",
    "R_factor_month",
    "full_dam",
    "dam_rem_cap",
    # NOTE: we KEEP E_t_ha_month (do NOT drop) so you still have avg erosion modulus
]

PCT_KEYS_FOR_PREV_YEAR = [
    "Erosion_pct",
    "Deposition_pct",
    "Vegetation_pct",
    "Reaction_pct",
    "River_lost_pct",
]

# ──────────────────────────────────────────────────────────────────────────────
# 2) Helper: compute annual means & pct metrics for one year & scenario
# ──────────────────────────────────────────────────────────────────────────────
def annual_summary_future(year: int, scenario: str):
    """
    Compute annual summaries for a given future year and scenario:
      - Read 12 monthly Parquet outputs from SOC_Future 7 / scenario
      - Filter out rows lacking Total_C
      - Compute annual mean of all numeric columns (means)
      - Derive:
          * Trapped_SOC_Dam (annual mean concentration in sedimentation area)
          * seven percentage metrics (mets, in %)
    Returns:
      - means: pd.Series of annual means for each numeric column (+ Trapped_SOC_Dam)
      - mets:  pd.Series of the seven percentage metrics (in %)
    """
    base_dir = OUTPUT_DIR / "Data" / "SOC_Future 7" / scenario

    monthly_dfs = []
    dam_ratios   = []  # % of Total_C in Region == sedimentation area
    low_ratios   = []  # % of Total_C in Low point == True
    dam_avg_concs = [] # average conc in sedimentation area

    for m in range(1, 13):
        fn = f"SOC_terms_{year}_{m:02d}_River.parquet"
        fp = base_dir / fn
        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp}")

        df = pd.read_parquet(fp)
        df = df[df["Total_C"].notna()]
        monthly_dfs.append(df)

        total_c = df["Total_C"].sum()
        if total_c > 0:
            dam_sum = df.loc[df["Region"] == "sedimentation area", "Total_C"].sum()
            low_sum = df.loc[df["Low point"] == "True", "Total_C"].sum()

            dam_ratios.append(dam_sum / total_c * 100.0)
            low_ratios.append(low_sum / total_c * 100.0)

            num_total = len(df)
            dam_avg_concs.append(dam_sum / num_total if num_total > 0 else 0.0)
        else:
            dam_ratios.append(0.0)
            low_ratios.append(0.0)
            dam_avg_concs.append(0.0)

    combined = pd.concat(monthly_dfs, ignore_index=True)
    combined = combined[combined["Total_C"].notna()]

    # Annual mean of all numeric columns
    means = combined.mean(numeric_only=True)

    # average concentration in sedimentation area (mean over months)
    means["Trapped_SOC_Dam"] = sum(dam_avg_concs) / len(dam_avg_concs)

    # five base % metrics using this year’s Total_C (will be adjusted later)
    erosion_pct    = (means["Erosion_fast"] + means["Erosion_slow"])       / means["Total_C"] * 100.0
    deposition_pct = (means["Deposition_fast"] + means["Deposition_slow"]) / means["Total_C"] * 100.0
    vegetation_pct = (means["Vegetation_fast"] + means["Vegetation_slow"]) / means["Total_C"] * 100.0
    reaction_pct   = (means["Reaction_fast"] + means["Reaction_slow"])     / means["Total_C"] * 100.0
    river_lost_pct =  means["Lost_SOC_River"]                              / means["Total_C"] * 100.0

    mets = pd.Series({
        "Erosion_pct":               erosion_pct,
        "Deposition_pct":            deposition_pct,
        "Vegetation_pct":            vegetation_pct,
        "Reaction_pct":              reaction_pct,
        "River_lost_pct":            river_lost_pct,
        "SOC_trapped_dam_pct":       sum(dam_ratios) / len(dam_ratios),
        "SOC_trapped_low_point_pct": sum(low_ratios) / len(low_ratios),
    })

    return means, mets


# ──────────────────────────────────────────────────────────────────────────────
# 3) Main: loop over scenarios and years; save one CSV per scenario
# ──────────────────────────────────────────────────────────────────────────────
def main():
    for scenario in SCENARIOS:
        print(f"\n=== Scenario {scenario} ===")

        years          = list(range(START_YEAR, END_YEAR + 1))
        means_per_year = {}
        mets_per_year  = {}

        # 3.1) First pass: compute raw means and mets
        for year in years:
            print(f"  Processing year {year} ...")
            means, mets = annual_summary_future(year, scenario)
            means_per_year[year] = means
            mets_per_year[year]  = mets

        # 3.2) Second pass: adjust 5 % metrics using previous-year Total_C
        adjusted_rows = []

        for i, year in enumerate(years):
            means = means_per_year[year]
            mets  = mets_per_year[year].copy()

            if i == 0:
                prev_total_c = means["Total_C"]  # first year uses its own
            else:
                prev_year    = years[i - 1]
                prev_total_c = means_per_year[prev_year]["Total_C"]

            for key in PCT_KEYS_FOR_PREV_YEAR:
                mets[key] = mets[key] * means["Total_C"] / prev_total_c

            # Drop unneeded columns from means, but KEEP E_t_ha_month
            means_clean = means.drop(labels=TO_DROP_MEANS, errors="ignore")

            # Build a wide row: year + all means + all pct metrics
            row = {"year": year}
            for k, v in means_clean.items():
                row[k] = v
            for k, v in mets.items():
                row[k] = v

            adjusted_rows.append(row)

        # 3.3) Build DataFrame and save for this scenario
        df_out = pd.DataFrame(adjusted_rows).sort_values("year")

        out_fp = OUTPUT_DIR / f"annual_future_summary_{scenario}_{START_YEAR}_{END_YEAR}.csv"
        df_out.to_csv(out_fp, index=False)

        print(f"  Saved scenario {scenario} summary to:\n    {out_fp}")

    print("\nAll scenarios processed.\n")


if __name__ == "__main__":
    main()
