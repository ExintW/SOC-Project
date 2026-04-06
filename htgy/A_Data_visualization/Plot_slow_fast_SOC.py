#!/usr/bin/env python3
"""
Plot monthly Fast/Slow SOC time series from SOC_terms_*_River.parquet outputs.

- Reads all matching parquet files in OUTPUT_DIR/Data
- Extracts Year/Month from filenames: SOC_terms_{year}_{month}_River.parquet
- Aggregates per-file (per-month) statistics over grid cells
- Plots monthly mean C_fast and C_slow over time
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import Paths 


# ======= CONFIG =======
DATA_DIR = Paths.OUTPUT_DIR / "Data"
FILE_GLOB = "SOC_terms_*_River.parquet"
# If you want to filter a year range, set these; otherwise leave None
YEAR_MIN: Optional[int] = 2000
YEAR_MAX: Optional[int] = 2014

# Aggregation method for each month: "mean" is usually what you want
AGG = "mean"  # "mean" | "sum" | "min" | "max"

# Optionally exclude river/lost cells or NaNs? (Your code likely sets rivers to 0 elsewhere,
# but in parquet they might appear as normal cells. We'll just ignore NaNs by default.)
# If you want to filter to some mask column, add it here.
# Example: only keep rows where Landuse != "river" (if you encoded it that way).
FILTER_EXPR = None  # e.g. 'Landuse != "river"'
# ======================


FNAME_RE = re.compile(r"SOC_terms_(\d{4})_(\d{2})_River\.parquet$")


def parse_year_month(path: str) -> Tuple[int, int]:
    base = os.path.basename(path)
    m = FNAME_RE.match(base)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {base}")
    year = int(m.group(1))
    month = int(m.group(2))
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month parsed from filename: {base}")
    return year, month


def reduce_series(x: np.ndarray, agg: str) -> float:
    # ignore NaNs
    if agg == "mean":
        return float(np.nanmean(x))
    if agg == "sum":
        return float(np.nansum(x))
    if agg == "min":
        return float(np.nanmin(x))
    if agg == "max":
        return float(np.nanmax(x))
    raise ValueError(f"Unknown AGG={agg}")


def main() -> None:
    pattern = os.path.join(DATA_DIR, FILE_GLOB)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No parquet files found at: {pattern}")

    rows = []

    for fp in files:
        year, month = parse_year_month(fp)

        if YEAR_MIN is not None and year < YEAR_MIN:
            continue
        if YEAR_MAX is not None and year > YEAR_MAX:
            continue

        df = pd.read_parquet(fp)

        # Basic sanity check
        for col in ("C_fast", "C_slow"):
            if col not in df.columns:
                raise SystemExit(f"Missing column '{col}' in {fp}")

        if FILTER_EXPR:
            df = df.query(FILTER_EXPR)

        c_fast = df["C_fast"].to_numpy(dtype=np.float64, copy=False)
        c_slow = df["C_slow"].to_numpy(dtype=np.float64, copy=False)

        rows.append(
            {
                "year": year,
                "month": month,
                "t": pd.Timestamp(year=year, month=month, day=1),
                "C_fast": reduce_series(c_fast, AGG),
                "C_slow": reduce_series(c_slow, AGG),
            }
        )

    if not rows:
        raise SystemExit("No files left after filtering (YEAR_MIN/YEAR_MAX/FILTER_EXPR).")

    ts = pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)

    # ---- Plot ----
    plt.figure()
    plt.plot(ts["t"], ts["C_fast"], label=f"C_fast ({AGG})")
    plt.plot(ts["t"], ts["C_slow"], label=f"C_slow ({AGG})")
    plt.xlabel("Month")
    plt.ylabel("SOC (g/kg)  [as stored in parquet]")
    plt.title("Monthly Fast vs Slow SOC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Paths.OUTPUT_DIR / "fast_slow_SOC_timeseries.png")
    plt.show()
    

    # Optional: print quick head/tail
    print(ts.head(5))
    print("...")
    print(ts.tail(5))


if __name__ == "__main__":
    main()
