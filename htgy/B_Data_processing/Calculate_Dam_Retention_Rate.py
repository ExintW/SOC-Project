#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Project globals (expects OUTPUT_DIR in globals.py; bulk density set below)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *   # OUTPUT_DIR, etc.
# ──────────────────────────────────────────────────────────────────────────────

YEARS = range(2008, 2020)  # inclusive 2008–2019

# Data locations
PRESENT_PARQUET_DIR = OUTPUT_DIR / "Data" / "SOC_Present 7"
STORAGE_PATH = OUTPUT_DIR / "dam_storage_by_year.csv"  # your yearly additions file

# Domain constants
AREA_KM2 = 640_000
AREA_HA  = AREA_KM2 * 100.0  # 1 km² = 100 ha
BULK_DENSITY = 1300          # kg/m^3 (same as your sim)

# Aggregation across rows per month for parquet
AGG_MODE = "mean"  # mean per-ha × total area


# ── Helpers ───────────────────────────────────────────────────────────────────

def convert_10k_m3_to_tons(val_10k_m3):
    """
    Convert '10k m^3' to tons (simulation-consistent):
      tons = val_10k_m3 * 10_000 * BULK_DENSITY / 1000
    """
    return np.asarray(val_10k_m3, dtype=float) * 10_000.0 * float(BULK_DENSITY) / 1000.0

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
    per_ha = vals.mean(skipna=True) if AGG_MODE == "mean" else vals.sum(skipna=True) / max(len(vals), 1)
    if pd.isna(per_ha):
        return 0.0
    return float(per_ha) * AREA_HA

def year_erosion_tons_parquet(root: Path, year: int) -> tuple[float, int]:
    total, months = 0.0, 0
    for _, fp in months_existing(root, year):
        total += month_erosion_tons_parquet(fp)
        months += 1
    return total, months

def dec_dam_sum_parquet(root: Path, year: int) -> Optional[float]:
    """
    Sum of 'dam_cur_stored' (tons) in December of `year`.
    Falls back to the latest available month if December missing.
    """
    dec_fp = parquet_path(root, year, 12)
    if dec_fp.exists():
        df = pd.read_parquet(dec_fp, columns=["dam_cur_stored"])
        return float(pd.to_numeric(df["dam_cur_stored"], errors="coerce").sum())

    latest_val, latest_m = None, None
    for m, fp in months_existing(root, year):
        df = pd.read_parquet(fp, columns=["dam_cur_stored"])
        val = float(pd.to_numeric(df["dam_cur_stored"], errors="coerce").sum())
        if latest_m is None or m > latest_m:
            latest_m, latest_val = m, val
    if latest_val is not None:
        print(f"[warn] {year}: December parquet missing; using month {latest_m} for dam_cur_stored.")
    return latest_val

def _read_storage_file(fp: Path) -> pd.DataFrame:
    """Try Excel first, then CSV; expect columns: year, total_storage_10k_m3."""
    try:
        return pd.read_excel(fp)
    except Exception:
        try:
            return pd.read_csv(fp)
        except Exception as e:
            raise FileNotFoundError(f"Unable to read storage file at {fp} as Excel or CSV. {e}")

def load_total_storage_tons_cumulative(storage_path_or_dir: Path, years: range) -> pd.Series:
    """
    Read table with:
      - year
      - total_storage_10k_m3 (newly added storage each year, in 10k m^3)
    Returns a Series(index=year) with cumulative storage in tons using the sim conversion.
    For years AFTER 2009, holds the cumulative at the 2009 level.
    """
    # resolve file
    if storage_path_or_dir.is_file():
        df = _read_storage_file(storage_path_or_dir)
        fp_used = storage_path_or_dir
    else:
        candidates = (
            list(storage_path_or_dir.glob("*.csv")) +
            list(storage_path_or_dir.glob("*.xlsx")) +
            list(storage_path_or_dir.glob("*.xls")) +
            [storage_path_or_dir]
        )
        df, fp_used = None, None
        for cand in candidates:
            if cand.exists() and cand.is_file():
                try:
                    df = _read_storage_file(cand)
                    fp_used = cand
                    break
                except Exception:
                    continue
        if df is None:
            raise FileNotFoundError(f"No readable storage file found at or under {storage_path_or_dir}")

    cols = {c.strip().lower(): c for c in df.columns}
    year_col = cols.get("year")
    add_col  = cols.get("total_storage_10k_m3")
    if year_col is None or add_col is None:
        raise KeyError(f"Storage file must have columns: 'year', 'total_storage_10k_m3' (file: {fp_used})")

    s = df[[year_col, add_col]].dropna()
    s[year_col] = pd.to_numeric(s[year_col], errors="coerce").astype("Int64")
    s[add_col]  = pd.to_numeric(s[add_col], errors="coerce")
    s = s.dropna().astype({year_col: int}).sort_values(year_col)

    # 10k m3 → tons, keep as Series so cumsum stays aligned to year
    added_tons = pd.Series(
        convert_10k_m3_to_tons(s[add_col].values),
        index=s[year_col].values, dtype=float
    ).sort_index()
    cum_tons = added_tons.cumsum()

    # clamp to 2009 for years after 2009
    clamp_year = 2009
    le_2009 = cum_tons[cum_tons.index <= clamp_year]
    clamp_value = float(le_2009.iloc[-1]) if not le_2009.empty else 0.0

    # build output over requested years
    out = {}
    for y in years:
        if y in cum_tons.index:
            out[y] = float(cum_tons.loc[y] if y <= clamp_year else clamp_value)
        elif y > clamp_year:
            out[y] = clamp_value
        else:
            # before first year in file or gaps → use previous value or 0
            prev_idx = [yy for yy in cum_tons.index if yy < y]
            out[y] = float(cum_tons.loc[max(prev_idx)]) if prev_idx else 0.0

    return pd.Series(out, name="total_storage_ton_cumulative").astype(float)


# ── Core computation ──────────────────────────────────────────────────────────

def compute_row(year: int) -> Optional[dict]:
    # Erosion (tons/year)
    erosion_ton_year, months_agg = year_erosion_tons_parquet(PRESENT_PARQUET_DIR, year)

    # Δ stored (tons) = Dec(curr) - Dec(prev) using dam_cur_stored
    dec_cur  = dec_dam_sum_parquet(PRESENT_PARQUET_DIR, year)
    dec_prev = dec_dam_sum_parquet(PRESENT_PARQUET_DIR, year - 1)

    if months_agg == 0:
        print(f"[skip] {year}: no monthly erosion data found.")
        return None

    dam_stored = np.nan
    if dec_cur is not None and dec_prev is not None:
        dam_stored = dec_cur - dec_prev

    retention = np.nan
    if erosion_ton_year and erosion_ton_year > 0 and not pd.isna(dam_stored):
        retention = dam_stored / erosion_ton_year

    return {
        "year": year,
        "period": "Present",
        "months_aggregated": months_agg,
        "erosion_ton_year": erosion_ton_year,      # tons/year
        "dam_stored_ton_year": dam_stored,         # tons/year
        "retention_rate": retention,               # unitless
    }


def main():
    # Annual rows
    rows = []
    for y in YEARS:
        row = compute_row(y)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No results produced; check inputs/paths/variables.")
        return

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    # Compute ratio using cumulative total storage (do NOT keep that column in final output)
    total_storage = load_total_storage_tons_cumulative(STORAGE_PATH, YEARS)  # Series indexed by year
    out["dam_stored_over_total_storage"] = out["dam_stored_ton_year"] / out["year"].map(total_storage)

    # Means over the window
    mean_retention = out["retention_rate"].mean(skipna=True)
    mean_ratio     = out["dam_stored_over_total_storage"].mean(skipna=True)

    # Append a single SUMMARY row (keeps everything in ONE file)
    summary_row = {
        "year": "SUMMARY",
        "period": "",
        "months_aggregated": np.nan,
        "erosion_ton_year": np.nan,
        "dam_stored_ton_year": np.nan,
        "retention_rate": np.nan,
        "dam_stored_over_total_storage": np.nan,
        "mean_retention_rate": mean_retention,
        "mean_dam_stored_over_total_storage": mean_ratio,
    }

    # Ensure the two mean columns exist for all rows (fill NaN), then append summary
    out["mean_retention_rate"] = np.nan
    out["mean_dam_stored_over_total_storage"] = np.nan
    out = pd.concat([out, pd.DataFrame([summary_row])], ignore_index=True)

    # Save just ONE file, without the total storage column (we never added it)
    out_fp = OUTPUT_DIR / "erosion_retention_2008_2019.csv"
    out.to_csv(out_fp, index=False)

    print(f"\nSaved annual+summary table → {out_fp}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
