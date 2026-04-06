#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Future SOC mean map visualization (2025 to 2100)
Style is CONSISTENT with your monthly plot_SOC_timestep() settings.

What this script does:
1) Reads monthly SOC parquet files for each future scenario (2025 to 2100)
2) Computes the time-mean SOC map (mean over all months in 2025 to 2100) at each grid cell
3) Saves 4 separate map figures (one per scenario)

Input:
OUTPUT_DIR / "Data" / "SOC_Future 7" / <scenario> / SOC_terms_YYYY_MM_River.parquet

Output:
OUTPUT_DIR / "Figure" / Mean_SOC_ssp126_2025_2100.png
OUTPUT_DIR / "Figure" / Mean_SOC_ssp245_2025_2100.png
OUTPUT_DIR / "Figure" / Mean_SOC_ssp370_2025_2100.png
OUTPUT_DIR / "Figure" / Mean_SOC_ssp585_2025_2100.png
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
import matplotlib.ticker as mticker

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================================
# (2) Globals and paths
# =============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import OUTPUT_DIR, DATA_DIR

FUTURE_DIR = OUTPUT_DIR / "Data" / "SOC_Future 7"
SCENARIOS = ["126", "245", "370", "585"]

START_YEAR = 2025
END_YEAR = 2100

# Loess Plateau border shapefile
LOESS_BORDER_PATH = Path(DATA_DIR) / "Loess_Plateau_vector_border.shp"

# Output folder
FIG_DIR = OUTPUT_DIR / "Figure"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# (3) Helpers: Parquet path and border plotting
# =============================================================================
def parquet_path(scen: str, year: int, month: int) -> Path:
    return FUTURE_DIR / scen / f"SOC_terms_{year}_{month:02d}_River.parquet"


def load_loess_border_boundary():
    border_gdf = gpd.read_file(LOESS_BORDER_PATH)
    geom = border_gdf.geometry.unary_union
    return geom.boundary


def plot_border(ax, boundary, linewidth=0.4):
    if isinstance(boundary, LineString):
        x, y = boundary.xy
        ax.plot(x, y, color="black", linewidth=linewidth)
    elif isinstance(boundary, MultiLineString):
        for seg in boundary.geoms:
            x, y = seg.xy
            ax.plot(x, y, color="black", linewidth=linewidth)


# =============================================================================
# (4) Helpers: Build stable grid index using one sample parquet
# =============================================================================
def find_any_existing_parquet_for_scenario(scen: str) -> Path:
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            fp = parquet_path(scen, year, month)
            if fp.exists():
                return fp
    raise FileNotFoundError(
        f"No parquet files found for scenario folder {scen}. "
        f"Check: {FUTURE_DIR / scen}"
    )


def build_base_index_from_parquet(sample_fp: Path):
    """
    Builds:
      - base_index: MultiIndex (lat, lon)
      - lon_sorted: sorted unique lon values
      - lat_sorted_desc: sorted unique lat values in descending order (for imshow)
    """
    df = pd.read_parquet(sample_fp)

    if ("LAT" not in df.columns) or ("LON" not in df.columns):
        raise KeyError("Parquet file missing LAT or LON columns.")

    df = df[["LAT", "LON"]].copy()
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df = df.dropna(subset=["LAT", "LON"])

    base_index = pd.MultiIndex.from_frame(
        df[["LAT", "LON"]].rename(columns={"LAT": "lat", "LON": "lon"})
    )
    base_index = base_index.drop_duplicates().sort_values()

    lon_sorted = np.sort(base_index.get_level_values("lon").unique().to_numpy())
    lat_sorted_desc = np.sort(base_index.get_level_values("lat").unique().to_numpy())[::-1]

    return base_index, lon_sorted, lat_sorted_desc


def align_totalc_to_base_index(df: pd.DataFrame, base_index: pd.MultiIndex) -> np.ndarray:
    """
    Align df["Total_C"] to base_index ordering (lat, lon).
    Returns a 1D numpy array of length len(base_index).
    """
    if "Total_C" not in df.columns:
        raise KeyError("Parquet missing Total_C column.")

    tmp = df[["LAT", "LON", "Total_C"]].copy()
    tmp["LAT"] = pd.to_numeric(tmp["LAT"], errors="coerce")
    tmp["LON"] = pd.to_numeric(tmp["LON"], errors="coerce")
    tmp["Total_C"] = pd.to_numeric(tmp["Total_C"], errors="coerce")
    tmp = tmp.dropna(subset=["LAT", "LON"])

    tmp = tmp.rename(columns={"LAT": "lat", "LON": "lon"}).set_index(["lat", "lon"])
    tmp = tmp[~tmp.index.duplicated(keep="first")]

    aligned = tmp.reindex(base_index)["Total_C"].to_numpy(dtype=float)
    return aligned


def reshape_to_grid(values_1d: np.ndarray,
                    base_index: pd.MultiIndex,
                    lon_sorted: np.ndarray,
                    lat_sorted_desc: np.ndarray) -> np.ndarray:
    """
    Convert aligned 1D values into a 2D grid for imshow.
    """
    s = pd.Series(values_1d, index=base_index)
    grid = s.unstack(level="lon")
    grid = grid.reindex(index=lat_sorted_desc, columns=lon_sorted)
    return grid.to_numpy(dtype=float)


# =============================================================================
# (5) Compute mean SOC map for one scenario (2025 to 2100)
# =============================================================================
def compute_future_mean_soc_map_for_scenario(scen: str,
                                            base_index: pd.MultiIndex) -> np.ndarray:
    """
    Returns:
      mean_soc_1d: aligned 1D SOC mean across all months (2025 to 2100)
    """
    soc_sum = np.zeros(len(base_index), dtype=float)
    soc_cnt = np.zeros(len(base_index), dtype=float)

    used_files = 0

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            fp = parquet_path(scen, year, month)
            if not fp.exists():
                continue

            df = pd.read_parquet(fp)
            aligned_soc = align_totalc_to_base_index(df, base_index)

            valid = np.isfinite(aligned_soc)
            soc_sum[valid] += aligned_soc[valid]
            soc_cnt[valid] += 1.0
            used_files += 1

        # progress print every 10 years
        if (year - START_YEAR) % 10 == 0:
            print(f"  ssp{scen}: finished year {year}")

    mean_soc = np.full(len(base_index), np.nan, dtype=float)
    ok = soc_cnt > 0
    mean_soc[ok] = soc_sum[ok] / soc_cnt[ok]

    print(f"ssp{scen}: used {used_files} monthly files")
    return mean_soc


# =============================================================================
# (6) Plot map with EXACT same settings as your monthly plot_SOC_timestep()
# =============================================================================
def plot_soc_map_consistent(grid2d: np.ndarray,
                            lon_sorted: np.ndarray,
                            lat_sorted_desc: np.ndarray,
                            title: str,
                            out_path: Path):
    boundary = load_loess_border_boundary()

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        grid2d,
        cmap="viridis",
        vmin=0, vmax=30,
        extent=[
            lon_sorted.min(), lon_sorted.max(),
            lat_sorted_desc.min(), lat_sorted_desc.max()
        ],
        origin="upper"
    )

    # Border outline (same linewidth)
    plot_border(ax, boundary, linewidth=0.4)

    # Colorbar axis: same height as map, same padding
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad="4%")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("SOC (g/kg)")

    # Labels and formatting (same)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="x")

    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =============================================================================
# (7) Main
# =============================================================================
def main():
    print("=" * 80)
    print("Future SOC mean maps (2025 to 2100), consistent plot style")
    print("=" * 80)

    # Build stable grid from a sample future parquet (ssp126)
    sample_fp = find_any_existing_parquet_for_scenario(SCENARIOS[0])
    base_index, lon_sorted, lat_sorted_desc = build_base_index_from_parquet(sample_fp)

    for scen in SCENARIOS:
        print("\n" + "=" * 80)
        print(f"Processing scenario: ssp{scen}")
        print("=" * 80)

        mean_soc_1d = compute_future_mean_soc_map_for_scenario(scen, base_index)
        grid_soc = reshape_to_grid(mean_soc_1d, base_index, lon_sorted, lat_sorted_desc)

        out_png = FIG_DIR / f"Mean_SOC_ssp{scen}_{START_YEAR}_{END_YEAR}.png"

        plot_soc_map_consistent(
            grid2d=grid_soc,
            lon_sorted=lon_sorted,
            lat_sorted_desc=lat_sorted_desc,
            title=f"Mean SOC from {START_YEAR} to {END_YEAR} for ssp{scen}",
            out_path=out_png
        )

    print("\nDone. 4 separate mean SOC maps created.")


if __name__ == "__main__":
    main()
