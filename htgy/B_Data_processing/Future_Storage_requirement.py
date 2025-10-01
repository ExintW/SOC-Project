#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Project globals (expects OUTPUT_DIR in globals.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *   # OUTPUT_DIR, etc.
# ──────────────────────────────────────────────────────────────────────────────

# Years & scenarios
YEARS = range(2025, 2101)                   # 2025–2100 inclusive
SSPS  = ["126", "245", "370", "585"]        # directory names under FUTURE_SOC_DIR

# Data locations
FUTURE_SOC_DIR = OUTPUT_DIR / "Data" / "SOC_Future 7"  # has subdirs 126/245/370/585 with parquet months
RETENTION_FILE = OUTPUT_DIR / "erosion_retention_2008_2019.csv"

# Domain constants (same as before)
AREA_KM2 = 640_000
AREA_HA  = AREA_KM2 * 100.0                 # 1 km² = 100 ha

# Aggregation across rows per month for parquet
AGG_MODE = "mean"  # mean per-ha × total area

# ── Helpers ───────────────────────────────────────────────────────────────────

def parquet_path(root: Path, year: int, month: int) -> Path:
    return root / f"SOC_terms_{year}_{month:02d}_River.parquet"

def months_existing(root: Path, year: int):
    for m in range(1, 13):
        fp = parquet_path(root, year, m)
        if fp.exists():
            yield m, fp

def month_erosion_tons_parquet(fp: Path) -> float:
    """
    tons/month from E_t_ha_month (tons/ha/month):
      tons/month = mean(E_t_ha_month) * total area (ha)
    """
    col = "E_t_ha_month"
    df = pd.read_parquet(fp, columns=[col])
    vals = pd.to_numeric(df[col], errors="coerce")

    if AGG_MODE == "sum":
        per_ha = vals.sum(skipna=True) / max(len(vals), 1)  # only if tiles are area-weighted
    else:
        per_ha = vals.mean(skipna=True)

    if pd.isna(per_ha):
        return 0.0
    return float(per_ha) * AREA_HA

def year_erosion_tons_parquet(root: Path, year: int) -> Tuple[float, int]:
    total, months = 0.0, 0
    for _, fp in months_existing(root, year):
        total += month_erosion_tons_parquet(fp)
        months += 1
    return total, months

def load_means_from_retention_file(fp: Path) -> Tuple[float, float]:
    """
    Load mean_retention_rate and mean_dam_stored_over_total_storage
    from the combined annual+summary CSV produced earlier.
    - If a SUMMARY row is present, read its two mean columns.
    - Otherwise, compute means over yearly rows.
    """
    if not fp.exists():
        raise FileNotFoundError(f"Retention table not found: {fp}")

    df = pd.read_csv(fp)

    # If there's a SUMMARY row, the means should already be present there
    if "year" in df.columns and df["year"].astype(str).str.upper().eq("SUMMARY").any():
        srow = df[df["year"].astype(str).str.upper() == "SUMMARY"].iloc[-1]
        mr = float(srow.get("mean_retention_rate", np.nan))
        mz = float(srow.get("mean_dam_stored_over_total_storage", np.nan))
        # If they are NaN (older file), compute from yearly rows
        if not np.isfinite(mr) or not np.isfinite(mz):
            yearly = df[~df["year"].astype(str).str.upper().eq("SUMMARY")]
            mr = float(pd.to_numeric(yearly["retention_rate"], errors="coerce").mean(skipna=True))
            mz = float(pd.to_numeric(yearly["dam_stored_over_total_storage"], errors="coerce").mean(skipna=True))
        return mr, mz

    # No explicit summary row → compute directly
    mr = float(pd.to_numeric(df["retention_rate"], errors="coerce").mean(skipna=True))
    mz = float(pd.to_numeric(df["dam_stored_over_total_storage"], errors="coerce").mean(skipna=True))
    return mr, mz

# ── Core computation ──────────────────────────────────────────────────────────

def compute_required_storage_for_scenario(ssp: str,
                                          mean_retention_rate: float,
                                          mean_dam_over_total: float) -> list[dict]:
    """
    For each year, compute erosion (tons/yr) and required total storage (tons):
      required_total_storage_ton = erosion_ton_year * mean_retention_rate / mean_dam_over_total
    """
    rows = []
    ssp_dir = FUTURE_SOC_DIR / ssp
    if not ssp_dir.exists():
        print(f"[warn] Future SOC dir missing for ssp{ssp}: {ssp_dir}")
        return rows

    for y in YEARS:
        erosion_ton_year, months_agg = year_erosion_tons_parquet(ssp_dir, y)
        if months_agg == 0:
            # Some datasets may start later or have gaps; skip year quietly
            continue

        # Guard against divide-by-zero
        required_storage = np.nan
        if mean_dam_over_total and np.isfinite(mean_dam_over_total):
            required_storage = erosion_ton_year * float(mean_retention_rate) / float(mean_dam_over_total)

        rows.append({
            "year": y,
            "scenario": f"ssp{ssp}",
            "months_aggregated": months_agg,
            "erosion_ton_year": erosion_ton_year,
            "required_total_storage_ton": required_storage,
            "mean_retention_rate": mean_retention_rate,
            "mean_dam_stored_over_total_storage": mean_dam_over_total,
        })
    return rows

def main():
    # 1) Load the two means from your 2008–2019 summary file
    mean_retention, mean_ratio = load_means_from_retention_file(RETENTION_FILE)
    if not np.isfinite(mean_retention) or not np.isfinite(mean_ratio) or mean_ratio == 0.0:
        raise ValueError(
            "Mean retention or mean dam_stored_over_total_storage is invalid. "
            f"Got retention={mean_retention}, ratio={mean_ratio} from {RETENTION_FILE}"
        )

    # 2) Compute per (year, scenario)
    all_rows = []
    for ssp in SSPS:
        all_rows.extend(
            compute_required_storage_for_scenario(ssp, mean_retention, mean_ratio)
        )

    if not all_rows:
        print("No results produced; check future SOC paths and filenames.")
        return

    out = pd.DataFrame(all_rows).sort_values(["scenario", "year"]).reset_index(drop=True)

    # 3) Save
    out_fp = OUTPUT_DIR / "required_total_storage_2025_2100_by_ssp.csv"
    out.to_csv(out_fp, index=False)
    print(f"\nSaved required total storage table → {out_fp}\n")
    print(out.head(20).to_string(index=False))  # show a peek

if __name__ == "__main__":
    main()
