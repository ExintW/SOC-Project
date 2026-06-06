"""
Plot_Source_Sink.py
====================
Result 3 — Spatiotemporal evolution of the SOC source / sink on the Loess Plateau
(黄土高原土壤有机碳源汇的时空演化过程).

Method:
    Treat each 1 km x 1 km cell as a GIS layer and overlay consecutive SOC
    states. Using the previous time step as the baseline, the per-cell change is

        dSOC(cell, t) = Total_C(cell, t) - Total_C(cell, t-1)

        dSOC < 0  ->  carbon SOURCE  (cell lost carbon)
        dSOC > 0  ->  carbon SINK    (cell gained carbon)

    To remove the strong intra-annual (quasi-sinusoidal) seasonality, the
    long-term source/sink trend is computed on ANNUAL means (year-over-year),
    while the per-cell classification is the standard raster map-algebra
    difference between two layers.

Outputs (matplotlib figures + CSV tables, no model re-run required):
    Figure/SourceSink_phase_maps.png   - mean annual dSOC per cell, by phase
    Figure/SourceSink_timeseries.png   - source/sink area % + net dSOC over time
    Annual_SourceSink_summary.csv      - per-year regional summary
    Phase_SourceSink_summary.csv       - per-phase regional summary

Note on attribution: a per-cell decomposition of dSOC into vegetation /
deposition / erosion / mineralization is NOT produced here. The stored monthly
snapshots do not let the signed fluxes close the per-cell mass balance exactly
(deposition routing and within-step ordering are not recoverable post hoc), so
any such panel would be misleading. Mechanistic driver attribution is already
handled rigorously by the segmented PLSR / VIP analysis in Result 2.

Run:
    uv run --active python htgy/A_Data_visualization/Plot_Source_Sink.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # OUTPUT_DIR, DATA_DIR

# =============================================================================
# (1) Configuration
# =============================================================================
# Location of the live model run. A single consistent 1950-2024 run writes its
# flat SOC_terms_*.parquet files here.
SRC_DIR = OUTPUT_DIR / "Data"
YEAR_START, YEAR_END = 1950, 2024

# Per-period directory mapping (first matching range wins; otherwise SRC_DIR is
# used). This mirrors the YEAR_DIRS convention in the repo's other plotting
# scripts and lets the analysis assemble a 1950-2024 series before a single
# combined run exists: the 2015-2024 future-extension years are read from the
# live run dir (SRC_DIR), while 1950-2014 falls back to the existing full
# reconstruction. Once one consistent 1950-2024 run has populated SRC_DIR,
# collapse this to a single (1950, 2024, SRC_DIR) entry so every year comes from
# the same run (avoids a model-version discontinuity at the 2014/2015 boundary).
YEAR_DIRS = [
    (1950, 2024, SRC_DIR),
]

# Three historical regimes from the manuscript (Fig. 1B / Fig. 2).
PHASES = [
    ("S1: 1950-1974 (degradation)", 1950, 1974),
    ("S2: 1975-1999 (conservation)", 1975, 1999),
    ("S3: 2000-2024 (restoration)", 2000, 2024),
]

# Sign threshold (g/kg/yr) below which a cell is treated as neutral, to avoid
# classifying numerical noise as a source or sink.
EPS = 0.01

LOESS_BORDER_PATH = Path(DATA_DIR) / "Loess_Plateau_vector_border.shp"
FIG_DIR = OUTPUT_DIR / "Figure"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# (2) Helpers: paths, grid index, border (mirrors the repo's other scripts)
# =============================================================================
def dir_for_year(year: int) -> Path:
    for y0, y1, d in YEAR_DIRS:
        if y0 <= year <= y1:
            return d
    return SRC_DIR


def parquet_path(year: int, month: int) -> Path:
    return dir_for_year(year) / f"SOC_terms_{year}_{month:02d}_River.parquet"


def build_base_index(sample_fp: Path):
    df = pd.read_parquet(sample_fp, columns=["LAT", "LON"])
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df = df.dropna(subset=["LAT", "LON"])
    base_index = pd.MultiIndex.from_frame(
        df.rename(columns={"LAT": "lat", "LON": "lon"})
    ).drop_duplicates().sort_values()
    lon_sorted = np.sort(base_index.get_level_values("lon").unique().to_numpy())
    lat_desc = np.sort(base_index.get_level_values("lat").unique().to_numpy())[::-1]
    return base_index, lon_sorted, lat_desc


def align(df: pd.DataFrame, base_index: pd.MultiIndex, col: str) -> np.ndarray:
    tmp = df[["LAT", "LON", col]].rename(columns={"LAT": "lat", "LON": "lon"})
    tmp = tmp.set_index(["lat", "lon"])
    tmp = tmp[~tmp.index.duplicated(keep="first")]
    return tmp.reindex(base_index)[col].to_numpy(dtype=float)


def reshape(values_1d, base_index, lon_sorted, lat_desc):
    s = pd.Series(values_1d, index=base_index).unstack(level="lon")
    return s.reindex(index=lat_desc, columns=lon_sorted).to_numpy(dtype=float)


def load_border_boundary():
    if not LOESS_BORDER_PATH.exists():
        return None
    geom = gpd.read_file(LOESS_BORDER_PATH).geometry.unary_union
    return geom.boundary


def plot_border(ax, boundary, lw=0.4):
    if boundary is None:
        return
    segs = boundary.geoms if isinstance(boundary, MultiLineString) else [boundary]
    for seg in segs:
        if isinstance(seg, LineString):
            x, y = seg.xy
            ax.plot(x, y, color="black", linewidth=lw)


# =============================================================================
# (3) Aggregate: annual-mean SOC grid + annual-sum flux grids per year
# =============================================================================
def aggregate_annual(base_index):
    """Returns, per year, the annual-mean Total_C (1D, aligned to base_index).
    Annual means remove the strong intra-annual seasonality before the
    year-over-year source/sink difference is taken."""
    soc_mean = {}
    for year in range(YEAR_START, YEAR_END + 1):
        soc_acc, n = None, 0
        for month in range(1, 13):
            fp = parquet_path(year, month)
            if not fp.exists():
                continue
            df = pd.read_parquet(fp, columns=["LAT", "LON", "Total_C"])
            soc = align(df, base_index, "Total_C")
            soc_acc = soc if soc_acc is None else np.nansum([soc_acc, soc], axis=0)
            n += 1
        if n == 0:
            continue
        soc_mean[year] = soc_acc / n
        print(f"  aggregated {year} ({n} months)")
    return soc_mean


# =============================================================================
# (4) Build per-year dSOC and regional summaries
# =============================================================================
def build_summary(soc_mean):
    years = sorted(soc_mean)
    rows = []
    dsoc_grids = {}
    for prev, cur in zip(years[:-1], years[1:]):
        if cur - prev != 1:
            continue
        d = soc_mean[cur] - soc_mean[prev]          # year-over-year dSOC, 1D
        dsoc_grids[cur] = d
        valid = ~np.isnan(d)
        nval = valid.sum()
        src = (d < -EPS) & valid
        snk = (d > EPS) & valid
        rows.append({
            "year": cur,
            "mean_Total_C": np.nanmean(soc_mean[cur]),
            "net_dSOC": np.nanmean(d),
            "source_area_pct": 100.0 * src.sum() / nval,
            "sink_area_pct": 100.0 * snk.sum() / nval,
            "source_mean_dSOC": np.nanmean(d[src]) if src.any() else np.nan,
            "sink_mean_dSOC": np.nanmean(d[snk]) if snk.any() else np.nan,
        })
    return pd.DataFrame(rows), dsoc_grids


def phase_mean_grid(dsoc_grids, y0, y1):
    arrs = [g for y, g in dsoc_grids.items() if y0 <= y <= y1]
    if not arrs:
        return None
    return np.nanmean(np.vstack(arrs), axis=0)


# =============================================================================
# (5) Figures
# =============================================================================
def fig_phase_maps(dsoc_grids, base_index, lon_sorted, lat_desc):
    boundary = load_border_boundary()
    vmax = max(
        np.nanpercentile(np.abs(phase_mean_grid(dsoc_grids, y0, y1)), 98)
        for _, y0, y1 in PHASES if phase_mean_grid(dsoc_grids, y0, y1) is not None
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(1, len(PHASES), figsize=(5.2 * len(PHASES), 5.0))
    extent = [lon_sorted.min(), lon_sorted.max(), lat_desc.min(), lat_desc.max()]
    im = None
    for ax, (label, y0, y1) in zip(axes, PHASES):
        pm = phase_mean_grid(dsoc_grids, y0, y1)
        if pm is None:
            ax.set_title(f"{label}\n(no data)", fontsize=11)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            continue
        grid = reshape(pm, base_index, lon_sorted, lat_desc)
        im = ax.imshow(grid, extent=extent, origin="upper", cmap="RdBu",
                       norm=norm, aspect="auto", interpolation="nearest")
        plot_border(ax, boundary)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Mean annual ΔSOC (g kg⁻¹ yr⁻¹)\nblue = sink, red = source")
    out = FIG_DIR / "SourceSink_phase_maps.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def fig_timeseries(summary):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.fill_between(summary.year, 0, summary.sink_area_pct,
                     color="#3b76af", alpha=0.5, label="Sink area %")
    ax1.fill_between(summary.year, 0, -summary.source_area_pct,
                     color="#c0392b", alpha=0.5, label="Source area %")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_ylabel("Grid area fraction (%)  sink ↑ / source ↓")
    ax1.set_xlabel("Year")
    ax2 = ax1.twinx()
    ax2.plot(summary.year, summary.net_dSOC, color="black", lw=1.8,
             label="Net regional ΔSOC")
    ax2.set_ylabel("Net regional ΔSOC (g kg⁻¹ yr⁻¹)")
    for _, y0, y1 in PHASES:
        ax1.axvline(y1 + 0.5, color="gray", ls="--", lw=0.6)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax1.set_title("Source / sink area and net SOC change, Loess Plateau")
    out = FIG_DIR / "SourceSink_timeseries.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


# =============================================================================
# (6) Main
# =============================================================================
def main():
    print(f"Source/sink analysis on: {SRC_DIR}")
    sample = parquet_path(YEAR_START, 1)
    if not sample.exists():
        raise FileNotFoundError(f"Sample parquet not found: {sample}")
    base_index, lon_sorted, lat_desc = build_base_index(sample)

    print("Aggregating annual grids...")
    soc_mean = aggregate_annual(base_index)

    print("Building summaries...")
    summary, dsoc_grids = build_summary(soc_mean)

    # Per-phase regional summary
    phase_rows = []
    for label, y0, y1 in PHASES:
        sub = summary[(summary.year >= y0) & (summary.year <= y1)]
        if sub.empty:
            continue
        phase_rows.append({
            "phase": label,
            "net_dSOC_mean": sub.net_dSOC.mean(),
            "source_area_pct_mean": sub.source_area_pct.mean(),
            "sink_area_pct_mean": sub.sink_area_pct.mean(),
        })
    phase_df = pd.DataFrame(phase_rows)

    summary_csv = OUTPUT_DIR / "Annual_SourceSink_summary.csv"
    phase_csv = OUTPUT_DIR / "Phase_SourceSink_summary.csv"
    summary.to_csv(summary_csv, index=False)
    phase_df.to_csv(phase_csv, index=False)
    print(f"  saved {summary_csv}")
    print(f"  saved {phase_csv}")
    print(phase_df.to_string(index=False))

    print("Rendering figures...")
    fig_phase_maps(dsoc_grids, base_index, lon_sorted, lat_desc)
    fig_timeseries(summary)
    print("Done.")


if __name__ == "__main__":
    main()
