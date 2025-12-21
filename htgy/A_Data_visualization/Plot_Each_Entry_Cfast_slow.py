#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot annual process-term time series (sum of fast + slow) for:
- One figure: 1950 to 2024 (Past and Present combined, if available)
- Four figures: 2025 to 2100 for scenarios 126, 245, 370, 585

Terms plotted:
- Vegetation  = Vegetation_fast + Vegetation_slow
- Reaction    = Reaction_fast + Reaction_slow
- Erosion     = Erosion_fast + Erosion_slow
- Deposition  = Deposition_fast + Deposition_slow

Additional series plotted in the same figure:
- C_fast
- C_slow

Axes:
- Left y-axis: Vegetation, Reaction, C_fast, C_slow
- Right y-axis: Erosion, Deposition (small magnitude)

Data sources (auto):
A) Future: reads OUTPUT_DIR/annual_future_summary_{scenario}_2025_2100.csv
B) Historical (1950 to 2024):
   1) If a precomputed annual CSV is found (see HIST_CANDIDATE_CSVS), it will use it
   2) Otherwise it tries to compute from monthly Parquet folders if they exist:
      - OUTPUT_DIR/Data/SOC_Past 2 (optional)
      - OUTPUT_DIR/Data/SOC_Present 7 (expected for 2007 to 2024 in your setup)
"""

# =============================================================================
# (1) Imports
# =============================================================================
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# (2) Project globals and paths
# =============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import OUTPUT_DIR

OUT_DIR = OUTPUT_DIR.joinpath("Process_Terms_Graphs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Future annual summary CSVs produced by your annual_future_summary script
FUTURE_SCENARIOS = ["126", "245", "370", "585"]
FUTURE_START_YEAR = 2025
FUTURE_END_YEAR = 2100

# For historical computation fallback (monthly Parquet)
PAST_TERMS_DIR = OUTPUT_DIR.joinpath("Data", "SOC_Past 2")       # optional
PRESENT_TERMS_DIR = OUTPUT_DIR.joinpath("Data", "SOC_Present 7") # expected
HIST_START_YEAR = 1950
HIST_END_YEAR = 2024

# If you already have an annual summary CSV for 1950 to 2024 terms, add it here
HIST_CANDIDATE_CSVS = [
    OUTPUT_DIR.joinpath("annual_soc_terms_overall_1950_2024.csv")
]

# Expected column names for fast+slow terms
FAST_SLOW_COLS = {
    "Erosion": ("Erosion_fast", "Erosion_slow"),
    "Deposition": ("Deposition_fast", "Deposition_slow"),
    "Vegetation": ("Vegetation_fast", "Vegetation_slow"),
    "Reaction": ("Reaction_fast", "Reaction_slow"),
}

# Additional SOC pool columns to plot
POOL_COLS = ["C_fast", "C_slow"]


# =============================================================================
# (3) Helpers
# =============================================================================
def _safe_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "year" not in df.columns:
        raise ValueError(f"CSV missing 'year' column: {path}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


def _build_terms_df_from_annual(df_annual: pd.DataFrame) -> pd.DataFrame:
    """
    Input df_annual must have:
      - year
      - fast/slow term columns
      - C_fast and C_slow
    Output DataFrame with columns:
      year, Erosion, Deposition, Vegetation, Reaction, C_fast, C_slow
    """
    out = pd.DataFrame({"year": df_annual["year"].astype(int)})

    # Build fast+slow sums
    for term, (c_fast, c_slow) in FAST_SLOW_COLS.items():
        if c_fast not in df_annual.columns or c_slow not in df_annual.columns:
            raise ValueError(f"Missing columns for {term}: need {c_fast} and {c_slow}")
        out[term] = pd.to_numeric(df_annual[c_fast], errors="coerce") + pd.to_numeric(df_annual[c_slow], errors="coerce")

    # Add SOC pools
    for c in POOL_COLS:
        if c not in df_annual.columns:
            raise ValueError(f"Missing SOC pool column in annual CSV: {c}")
        out[c] = pd.to_numeric(df_annual[c], errors="coerce")

    out = out.sort_values("year").reset_index(drop=True)
    return out


def compute_annual_terms_from_monthly_parquet(base_dir: Path, year_start: int, year_end: int) -> pd.DataFrame:
    """
    Compute annual mean by reading 12 monthly Parquet files per year.

    File pattern:
      SOC_terms_{year}_{month:02d}_River.parquet

    Returns DataFrame with:
      year, Erosion, Deposition, Vegetation, Reaction, C_fast, C_slow
    """
    rows = []

    for year in range(year_start, year_end + 1):
        monthly = []

        for m in range(1, 13):
            fp = base_dir.joinpath(f"SOC_terms_{year}_{m:02d}_River.parquet")
            if not fp.exists():
                continue

            dfm = pd.read_parquet(fp)
            # Keep only valid SOC rows
            if "Total_C" in dfm.columns:
                dfm = dfm[dfm["Total_C"].notna()]
            monthly.append(dfm)

        if len(monthly) == 0:
            continue

        combined = pd.concat(monthly, ignore_index=True)
        means = combined.mean(numeric_only=True)

        row = {"year": year}

        # fast+slow terms
        ok = True
        for term, (c_fast, c_slow) in FAST_SLOW_COLS.items():
            if (c_fast not in means.index) or (c_slow not in means.index):
                ok = False
                break
            row[term] = float(means[c_fast] + means[c_slow])

        # C_fast and C_slow pools
        for c in POOL_COLS:
            if c not in means.index:
                ok = False
                break
            row[c] = float(means[c])

        if ok:
            rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame(columns=["year"] + list(FAST_SLOW_COLS.keys()) + POOL_COLS)

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def plot_terms(df_terms: pd.DataFrame, title: str, out_path: Path):
    """
    Plot annual process terms with two y-axes (different scales):
      - Left axis: Vegetation, Reaction, C_fast, C_slow
      - Right axis: Erosion, Deposition
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x = df_terms["year"].values

    # Left axis: large-scale series
    left_series = ["Vegetation", "Reaction", "C_fast", "C_slow"]
    left_lines, left_labels = [], []

    for name in left_series:
        if name in df_terms.columns:
            line, = ax1.plot(
                x,
                df_terms[name].values,
                label=name,
                linewidth=1.6
            )
            left_lines.append(line)
            left_labels.append(name)

    # Right axis: small-scale series (dashed)
    right_series = ["Erosion", "Deposition"]
    right_lines, right_labels = [], []

    for name in right_series:
        if name in df_terms.columns:
            line, = ax2.plot(
                x,
                df_terms[name].values,
                label=name,
                linewidth=1.6,
                linestyle="--"
            )
            right_lines.append(line)
            right_labels.append(name)

    # Labels and formatting
    ax1.set_title(title)
    ax1.set_xlabel("Year")

    ax1.set_ylabel("Annual mean [Vegetation, Reaction, C_fast, C_slow] (g/kg/month)")
    ax2.set_ylabel("Annual mean [Erosion, Deposition] (g/kg/month)")

    ax1.grid(True, alpha=0.3)

    # Combine legend (both axes)
    lines = left_lines + right_lines
    labels = left_labels + right_labels
    ax1.legend(lines, labels, loc="upper left")

    # Decade ticks
    if len(x) > 0:
        start = int(np.nanmin(x))
        end = int(np.nanmax(x))
        ticks = list(range((start // 10) * 10, end + 1, 10))
        ax1.set_xticks(ticks)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =============================================================================
# (4) Load or compute historical (1950 to 2024)
# =============================================================================
def load_or_compute_hist_terms() -> pd.DataFrame:
    # Try precomputed annual CSV first
    for cand in HIST_CANDIDATE_CSVS:
        if cand.exists():
            df_annual = _safe_read_csv(cand)
            df_terms = _build_terms_df_from_annual(df_annual)
            df_terms = df_terms[(df_terms["year"] >= HIST_START_YEAR) & (df_terms["year"] <= HIST_END_YEAR)].copy()
            return df_terms

    # Otherwise compute from monthly Parquet folders if they exist
    parts = []

    if PAST_TERMS_DIR.exists():
        df_past = compute_annual_terms_from_monthly_parquet(PAST_TERMS_DIR, HIST_START_YEAR, 2006)
        parts.append(df_past)

    if PRESENT_TERMS_DIR.exists():
        df_pres = compute_annual_terms_from_monthly_parquet(PRESENT_TERMS_DIR, 2007, HIST_END_YEAR)
        parts.append(df_pres)

    if len(parts) == 0:
        raise FileNotFoundError(
            "No historical term source found. "
            "Either provide a precomputed annual CSV in HIST_CANDIDATE_CSVS "
            "or ensure monthly Parquet folders exist for past and present."
        )

    df_all = pd.concat(parts, ignore_index=True).sort_values("year").reset_index(drop=True)
    df_all = df_all[(df_all["year"] >= HIST_START_YEAR) & (df_all["year"] <= HIST_END_YEAR)].copy()
    return df_all


# =============================================================================
# (5) Load future (2025 to 2100) from annual summary CSVs
# =============================================================================
def load_future_terms(scenario: str) -> pd.DataFrame:
    fp = OUTPUT_DIR.joinpath(f"annual_future_summary_{scenario}_{FUTURE_START_YEAR}_{FUTURE_END_YEAR}.csv")
    if not fp.exists():
        raise FileNotFoundError(f"Missing future annual summary CSV: {fp}")

    df_annual = _safe_read_csv(fp)
    df_terms = _build_terms_df_from_annual(df_annual)
    df_terms = df_terms[(df_terms["year"] >= FUTURE_START_YEAR) & (df_terms["year"] <= FUTURE_END_YEAR)].copy()
    return df_terms


# =============================================================================
# (6) Main
# =============================================================================
def main():
    # Historical plot
    hist_terms = load_or_compute_hist_terms()
    out_hist = OUT_DIR.joinpath("process_terms_with_Cfast_Cslow_1950_2024.png")
    plot_terms(
        hist_terms,
        "Annual process terms + SOC pools (1950 to 2024)",
        out_hist
    )
    print(f"Saved: {out_hist}")

    # Future plots, one per scenario
    for scen in FUTURE_SCENARIOS:
        df_terms = load_future_terms(scen)
        out_fp = OUT_DIR.joinpath(f"process_terms_with_Cfast_Cslow_ssp{scen}_2025_2100.png")
        plot_terms(
            df_terms,
            f"Annual process terms + SOC pools SSP{scen} (2025 to 2100)",
            out_fp
        )
        print(f"Saved: {out_fp}")

    print(f"All figures saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
