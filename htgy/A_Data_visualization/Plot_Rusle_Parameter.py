#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot annual mean RUSLE parameter time series for:
- One figure: 1950 to 2024
- Four figures: 2025 to 2100 for scenarios 126, 245, 370, 585

Parameters plotted:
- Left y-axis: R factor, LS factor
- Right y-axis: C factor, K factor, P factor

Right y-axis range is fixed to 0 to 1.2.

Data source:
Reads monthly Parquet files with names:
    SOC_terms_{year}_{month:02d}_River.parquet

Required Parquet columns:
- C_factor_month
- K_factor_month
- LS_factor_month
- P_factor_month
- R_factor_month
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

OUT_DIR = OUTPUT_DIR.joinpath("RUSLE_Factor_Graphs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Historical
PAST_DIR = OUTPUT_DIR.joinpath("Data", "SOC_Past 2")
PRESENT_DIR = OUTPUT_DIR.joinpath("Data", "SOC_Present 7")
HIST_START_YEAR = 1950
HIST_END_YEAR = 2024

# Future
FUTURE_SCENARIOS = ["126", "245", "370", "585"]
FUTURE_START_YEAR = 2025
FUTURE_END_YEAR = 2100
FUTURE_BASE_DIR = OUTPUT_DIR.joinpath("Data", "SOC_Future 7")
# If needed, change to:
# FUTURE_BASE_DIR = OUTPUT_DIR.joinpath("Data", "SOC_Future 4")

# Parquet columns
RUSLE_COLS = {
    "C factor": "C_factor_month",
    "K factor": "K_factor_month",
    "LS factor": "LS_factor_month",
    "P factor": "P_factor_month",
    "R factor": "R_factor_month",
}


# =============================================================================
# (3) Helpers
# =============================================================================
def compute_annual_rusle_from_monthly_parquet(base_dir: Path, year_start: int, year_end: int) -> pd.DataFrame:
    """
    Compute annual mean RUSLE parameters from monthly Parquet files.

    File pattern:
        SOC_terms_{year}_{month:02d}_River.parquet

    Returns:
        DataFrame with columns:
        year, C factor, K factor, LS factor, P factor, R factor
    """
    rows = []

    for year in range(year_start, year_end + 1):
        monthly_frames = []

        for month in range(1, 13):
            fp = base_dir.joinpath(f"SOC_terms_{year}_{month:02d}_River.parquet")
            if not fp.exists():
                continue

            dfm = pd.read_parquet(fp)

            # Keep same style as your other scripts
            if "Total_C" in dfm.columns:
                dfm = dfm[dfm["Total_C"].notna()]

            monthly_frames.append(dfm)

        if len(monthly_frames) == 0:
            continue

        combined = pd.concat(monthly_frames, ignore_index=True)
        means = combined.mean(numeric_only=True)

        row = {"year": year}
        ok = True

        for label, col in RUSLE_COLS.items():
            if col not in means.index:
                ok = False
                print(f"[WARNING] Missing column '{col}' for year {year} in {base_dir}")
                break
            row[label] = float(means[col])

        if ok:
            rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame(columns=["year"] + list(RUSLE_COLS.keys()))

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def plot_rusle(df_annual: pd.DataFrame, title: str, out_path: Path,
               ylim_left=None, ylim_right=(0, 1.2)):
    """
    Plot annual mean RUSLE parameters with two y-axes:
      - Left axis: R factor, LS factor
      - Right axis: C factor, K factor, P factor
    """

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x = df_annual["year"].values

    # -------------------------------------------------------------------------
    # Custom styles
    # -------------------------------------------------------------------------
    left_style = {
        "R factor":  {"color": "tab:blue",   "linestyle": "-",  "linewidth": 2.2},
        "LS factor": {"color": "tab:orange", "linestyle": "--", "linewidth": 2.2},
    }

    right_style = {
        "C factor": {"color": "tab:green",  "linestyle": "-",  "linewidth": 2.2},
        "K factor": {"color": "tab:red",    "linestyle": "--", "linewidth": 2.2},
        "P factor": {"color": "tab:purple", "linestyle": "-.", "linewidth": 2.2},
    }

    # -------------------------------------------------------------------------
    # Left axis: R and LS
    # -------------------------------------------------------------------------
    left_terms = ["R factor", "LS factor"]
    left_lines = []
    left_labels = []

    for term in left_terms:
        if term in df_annual.columns:
            style = left_style[term]
            line, = ax1.plot(
                x,
                df_annual[term].values,
                label=term,
                **style
            )
            left_lines.append(line)
            left_labels.append(term)

    # -------------------------------------------------------------------------
    # Right axis: C, K, P
    # -------------------------------------------------------------------------
    right_terms = ["C factor", "K factor", "P factor"]
    right_lines = []
    right_labels = []

    for term in right_terms:
        if term in df_annual.columns:
            style = right_style[term]
            line, = ax2.plot(
                x,
                df_annual[term].values,
                label=term,
                **style
            )
            right_lines.append(line)
            right_labels.append(term)

    # -------------------------------------------------------------------------
    # Labels, limits, ticks
    # -------------------------------------------------------------------------
    ax1.set_title(title, fontsize=18)
    ax1.set_xlabel("Year", fontsize=14)
    ax1.set_ylabel("Annual mean value [R factor, LS factor]", fontsize=14)
    ax2.set_ylabel("Annual mean value [C factor, K factor, P factor]", fontsize=14)

    ax1.tick_params(axis="both", labelsize=12, length=6, width=1.2)
    ax2.tick_params(axis="y", labelsize=12, length=6, width=1.2)

    if ylim_left is not None:
        ax1.set_ylim(ylim_left[0], ylim_left[1])

    if ylim_right is not None:
        ax2.set_ylim(ylim_right[0], ylim_right[1])

    # -------------------------------------------------------------------------
    # Grid and legend
    # -------------------------------------------------------------------------
    ax1.grid(True, alpha=0.3)

    lines = left_lines + right_lines
    labels = left_labels + right_labels
    ax1.legend(lines, labels, loc="upper right", fontsize=12)

    # -------------------------------------------------------------------------
    # Decade ticks
    # -------------------------------------------------------------------------
    if len(x) > 0:
        start = int(np.nanmin(x))
        end = int(np.nanmax(x))
        ticks = list(range((start // 10) * 10, end + 1, 10))
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(ticks, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def load_historical_rusle() -> pd.DataFrame:
    """
    Load historical annual mean RUSLE parameters by combining:
    - SOC_Past 2: 1950 to 2006
    - SOC_Present 7: 2007 to 2024
    """
    parts = []

    if PAST_DIR.exists():
        df_past = compute_annual_rusle_from_monthly_parquet(PAST_DIR, HIST_START_YEAR, 2006)
        parts.append(df_past)
    else:
        print(f"[WARNING] Historical past directory not found: {PAST_DIR}")

    if PRESENT_DIR.exists():
        df_present = compute_annual_rusle_from_monthly_parquet(PRESENT_DIR, 2007, HIST_END_YEAR)
        parts.append(df_present)
    else:
        print(f"[WARNING] Historical present directory not found: {PRESENT_DIR}")

    if len(parts) == 0:
        raise FileNotFoundError("No historical Parquet folders found.")

    df_all = pd.concat(parts, ignore_index=True)
    df_all = df_all.sort_values("year").drop_duplicates(subset=["year"]).reset_index(drop=True)
    df_all = df_all[(df_all["year"] >= HIST_START_YEAR) & (df_all["year"] <= HIST_END_YEAR)].copy()

    return df_all


def load_future_rusle(scenario: str) -> pd.DataFrame:
    """
    Load future annual mean RUSLE parameters from:
    OUTPUT_DIR/Data/SOC_Future 7/{scenario}
    """
    scen_dir = FUTURE_BASE_DIR.joinpath(scenario)

    if not scen_dir.exists():
        raise FileNotFoundError(f"Future scenario directory not found: {scen_dir}")

    df = compute_annual_rusle_from_monthly_parquet(
        scen_dir,
        FUTURE_START_YEAR,
        FUTURE_END_YEAR
    )

    return df


# =============================================================================
# (4) Main
# =============================================================================
def main():
    YLIM_CFG = {
        "hist": {"left": None, "right": (0, 1.2)},
        "126": {"left": None, "right": (0, 1.2)},
        "245": {"left": None, "right": (0, 1.2)},
        "370": {"left": None, "right": (0, 1.2)},
        "585": {"left": None, "right": (0, 1.2)},
    }

    # -------------------------------------------------------------------------
    # 4.1 Historical plot
    # -------------------------------------------------------------------------
    hist_df = load_historical_rusle()

    out_hist_csv = OUT_DIR.joinpath("annual_rusle_factors_1950_2024.csv")
    out_hist_fig = OUT_DIR.joinpath("annual_rusle_factors_1950_2024.png")

    hist_df.to_csv(out_hist_csv, index=False)
    print(f"Saved: {out_hist_csv}")

    cfg = YLIM_CFG["hist"]
    plot_rusle(
        hist_df,
        title="Annual mean RUSLE factors 1950 to 2024",
        out_path=out_hist_fig,
        ylim_left=cfg["left"],
        ylim_right=cfg["right"]
    )
    print(f"Saved: {out_hist_fig}")

    # -------------------------------------------------------------------------
    # 4.2 Future plots, one per scenario
    # -------------------------------------------------------------------------
    for scen in FUTURE_SCENARIOS:
        df_future = load_future_rusle(scen)

        out_csv = OUT_DIR.joinpath(f"annual_rusle_factors_ssp{scen}_2025_2100.csv")
        out_fig = OUT_DIR.joinpath(f"annual_rusle_factors_ssp{scen}_2025_2100.png")

        df_future.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        cfg = YLIM_CFG.get(scen, {"left": None, "right": (0, 1.2)})
        plot_rusle(
            df_future,
            title=f"Annual mean RUSLE factors SSP{scen}, 2025 to 2100",
            out_path=out_fig,
            ylim_left=cfg["left"],
            ylim_right=cfg["right"]
        )
        print(f"Saved: {out_fig}")

    print(f"All outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()