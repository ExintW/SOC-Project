#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize:
1) Mean LAI map (1950–2024) from NetCDF (CMIP6 resampled point LAI)
2) Mean SOC map (1950–2024 average) for cells where mean LAI > 1.231
   SOC is read from monthly Parquet (SOC_terms_YYYY_MM_River.parquet)
3) Also saves annual mean SOC over high-LAI cells to CSV (optional)

Why this version:
- Your Parquet SOC files do NOT contain LAI, so LAI must come from NetCDF.

Outputs:
- OUTPUT_DIR/Figure/Mean_LAI_1950_2024.png
- OUTPUT_DIR/Figure/Mean_SOC_highLAI_1950_2024.png
- OUTPUT_DIR/Annual_SOC_highLAI_1950_2024.csv
"""

# =============================================================================
# (1) Imports
# =============================================================================
import os
import sys
from pathlib import Path
from scipy.spatial import cKDTree


import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================================
# (2) Globals, paths, configuration
# =============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import OUTPUT_DIR, DATA_DIR, PROCESSED_DIR

START_YEAR = 1950
END_YEAR   = 2024
LAI_THRESHOLD = 1.231

# --- LAI NetCDF files (same naming as your LAI script) ---
LAI_FILES = {
    "historical": "resampled_lai_points_1950-2000.nc",
    "present":    "resampled_lai_points_2001-2014.nc",
    "ssp245":     "resampled_lai_points_2015-2100_245.nc",  # for 2015–2024
}
LAI_NC_DIR = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled"

# --- SOC monthly Parquet folders (edit if needed) ---
YEAR_DIRS = [
    (1950, 2006, OUTPUT_DIR / "Data" / "SOC_Past 2"),      # change if you have a different past folder
    (2007, 2024, OUTPUT_DIR / "Data" / "SOC_Present 7"),
]

# Loess Plateau border shapefile
LOESS_BORDER_PATH = Path(DATA_DIR) / "Loess_Plateau_vector_border.shp"

# Output
FIG_DIR = OUTPUT_DIR / "Figure"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_LAI_MAP = FIG_DIR / "Mean_LAI_1950_2024.png"
OUT_SOC_MAP = FIG_DIR / "Mean_SOC_highLAI_1950_2024.png"
OUT_SOC_CSV = OUTPUT_DIR / "Annual_SOC_highLAI_1950_2024.csv"


# =============================================================================
# (3) Helpers: SOC parquet paths
# =============================================================================
def find_base_dir_for_year(year: int) -> Path:
    for y0, y1, d in YEAR_DIRS:
        if y0 <= year <= y1:
            return Path(d)
    raise FileNotFoundError(f"No directory mapping found for year {year}. Update YEAR_DIRS.")

def parquet_path(year: int, month: int) -> Path:
    return find_base_dir_for_year(year) / f"SOC_terms_{year}_{month:02d}_River.parquet"


# =============================================================================
# (4) Helpers: border plotting (your format)
# =============================================================================
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
# (5) Helpers: build a stable point index on (lat, lon)
# =============================================================================
def build_base_index_from_soc_parquet(sample_fp: Path):
    df = pd.read_parquet(sample_fp)
    if "LAT" not in df.columns or "LON" not in df.columns:
        raise KeyError("SOC parquet missing LAT/LON columns.")
    df = df[["LAT", "LON"]].copy()
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df = df.dropna(subset=["LAT", "LON"])

    # stable MultiIndex sorted by (LAT, LON)
    base_index = pd.MultiIndex.from_frame(df[["LAT", "LON"]].rename(columns={"LAT":"lat", "LON":"lon"}))
    base_index = base_index.drop_duplicates().sort_values()

    lon_sorted = np.sort(base_index.get_level_values("lon").unique().to_numpy())
    lat_sorted_desc = np.sort(base_index.get_level_values("lat").unique().to_numpy())[::-1]
    return base_index, lon_sorted, lat_sorted_desc


def align_series_to_base_index(df: pd.DataFrame, base_index: pd.MultiIndex, value_col: str) -> np.ndarray:
    tmp = df[["LAT","LON", value_col]].copy()
    tmp["LAT"] = pd.to_numeric(tmp["LAT"], errors="coerce")
    tmp["LON"] = pd.to_numeric(tmp["LON"], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=["LAT","LON"])

    tmp = tmp.rename(columns={"LAT":"lat","LON":"lon"}).set_index(["lat","lon"])
    tmp = tmp[~tmp.index.duplicated(keep="first")]
    aligned = tmp.reindex(base_index)[value_col].to_numpy(dtype=float)
    return aligned


def reshape_to_grid(values_1d: np.ndarray, base_index: pd.MultiIndex,
                    lon_sorted: np.ndarray, lat_sorted_desc: np.ndarray) -> np.ndarray:
    s = pd.Series(values_1d, index=base_index)
    grid = s.unstack(level="lon")
    grid = grid.reindex(index=lat_sorted_desc, columns=lon_sorted)
    return grid.to_numpy(dtype=float)


# =============================================================================
# (6) Read LAI from NetCDF and align to SOC grid by lon/lat
# =============================================================================
def read_lai_points_mean_1950_2024() -> pd.DataFrame:
    """
    Returns a DataFrame with columns: lon, lat, mean_lai
    Mean computed over all months within 1950–2024:
      historical 1950–2000
      present    2001–2014
      ssp245     2015–2024
    """
    parts = []

    # historical
    fp = LAI_NC_DIR / LAI_FILES["historical"]
    ds = xr.open_dataset(fp)
    lai = ds["lai"].sel(time=slice("1950-01-01", "2000-12-31")).mean(dim="time")
    parts.append(pd.DataFrame({"lon": ds["lon"].values, "lat": ds["lat"].values, "lai": lai.values}))
    ds.close()

    # present
    fp = LAI_NC_DIR / LAI_FILES["present"]
    ds = xr.open_dataset(fp)
    lai = ds["lai"].sel(time=slice("2001-01-01", "2014-12-31")).mean(dim="time")
    parts.append(pd.DataFrame({"lon": ds["lon"].values, "lat": ds["lat"].values, "lai": lai.values}))
    ds.close()

    # 2015–2024 from ssp245 (you can change scenario if you prefer)
    fp = LAI_NC_DIR / LAI_FILES["ssp245"]
    ds = xr.open_dataset(fp)
    lai = ds["lai"].sel(time=slice("2015-01-01", "2024-12-31")).mean(dim="time")
    parts.append(pd.DataFrame({"lon": ds["lon"].values, "lat": ds["lat"].values, "lai": lai.values}))
    ds.close()

    df = pd.concat(parts, ignore_index=True)

    # If the point set is identical across files, mean of concatenated is fine,
    # but to be safe: group by lon/lat and average.
    df["lai"] = pd.to_numeric(df["lai"], errors="coerce")
    out = df.groupby(["lat","lon"], as_index=False)["lai"].mean().rename(columns={"lai":"mean_lai"})
    return out


def align_lai_to_soc_grid(mean_lai_df: pd.DataFrame, base_index: pd.MultiIndex, tol: float = 1e-4) -> np.ndarray:
    """
    Align mean LAI (from NetCDF point list) to the SOC grid base_index (lat, lon)
    using nearest-neighbor matching (KDTree), which avoids MultiIndex duplicate issues.

    tol is in degrees. 1e-4 is usually safe for tiny floating mismatches.
    """
    df = mean_lai_df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["mean_lai"] = pd.to_numeric(df["mean_lai"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "mean_lai"])

    # Ensure unique LAI points (drop exact duplicates)
    df = df.drop_duplicates(subset=["lat", "lon"], keep="first")

    # LAI coordinates for KDTree
    lai_xy = np.column_stack([df["lat"].values, df["lon"].values])
    tree = cKDTree(lai_xy)

    # SOC coordinates from base_index
    soc_lat = base_index.get_level_values("lat").to_numpy(dtype=float)
    soc_lon = base_index.get_level_values("lon").to_numpy(dtype=float)
    soc_xy = np.column_stack([soc_lat, soc_lon])

    # Nearest neighbor match
    dist, idx = tree.query(soc_xy, k=1)

    aligned = np.full(len(base_index), np.nan, dtype=float)
    ok = dist <= tol
    aligned[ok] = df["mean_lai"].values[idx[ok]]

    # Optional: print match rate so you can sanity check
    match_rate = ok.mean() * 100.0
    print(f"LAI alignment: matched {match_rate:.1f}% of SOC points within tol={tol}")

    return aligned

# =============================================================================
# (7) Compute SOC statistics for high-LAI cells
# =============================================================================
def detect_soc_series(df: pd.DataFrame) -> pd.Series:
    if "Total_C" in df.columns:
        return pd.to_numeric(df["Total_C"], errors="coerce")
    if ("C_fast" in df.columns) and ("C_slow" in df.columns):
        return pd.to_numeric(df["C_fast"], errors="coerce") + pd.to_numeric(df["C_slow"], errors="coerce")
    raise KeyError("SOC parquet needs Total_C or (C_fast and C_slow).")


def compute_soc_mean_map_and_annual(base_index: pd.MultiIndex, high_mask: np.ndarray):
    """
    Returns:
      - mean_soc_1d_high: 1D mean SOC over 1950–2024, but only for high-LAI cells (others NaN)
      - annual_df: DataFrame year, soc_mean_high_lai
    """
    soc_sum = np.zeros(len(base_index), dtype=float)
    soc_cnt = np.zeros(len(base_index), dtype=float)

    annual_rows = []

    for year in range(START_YEAR, END_YEAR + 1):
        y_sum = 0.0
        y_cnt = 0.0

        for m in range(1, 13):
            fp = parquet_path(year, m)
            if not fp.exists():
                continue

            df = pd.read_parquet(fp)
            df = df[df["Total_C"].notna()] if "Total_C" in df.columns else df

            df = df.copy()
            df["_SOC_"] = detect_soc_series(df)

            aligned_soc = align_series_to_base_index(df, base_index, "_SOC_")

            # accumulate overall mean SOC map
            valid = np.isfinite(aligned_soc)
            soc_sum[valid] += aligned_soc[valid]
            soc_cnt[valid] += 1.0

            # annual mean over high-LAI cells
            v = aligned_soc[high_mask]
            v = v[np.isfinite(v)]
            y_sum += float(np.nansum(v))
            y_cnt += float(np.sum(np.isfinite(v)))

        annual_rows.append({
            "year": year,
            "soc_mean_high_lai": (y_sum / y_cnt) if y_cnt > 0 else np.nan
        })

    mean_soc = np.full(len(base_index), np.nan, dtype=float)
    ok = soc_cnt > 0
    mean_soc[ok] = soc_sum[ok] / soc_cnt[ok]

    mean_soc_high = np.full(len(base_index), np.nan, dtype=float)
    mean_soc_high[high_mask] = mean_soc[high_mask]

    return mean_soc_high, pd.DataFrame(annual_rows)


# =============================================================================
# (8) Plot maps (your requested format)
# =============================================================================
def plot_map_imshow(grid2d: np.ndarray, lon_sorted: np.ndarray, lat_sorted_desc: np.ndarray,
                    title: str, cbar_label: str, out_path: Path,
                    cmap="viridis", vmin=None, vmax=None):
    boundary = load_loess_border_boundary()

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        grid2d,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=[lon_sorted.min(), lon_sorted.max(), lat_sorted_desc.min(), lat_sorted_desc.max()],
        origin="upper"
    )

    # border outline
    plot_border(ax, boundary, linewidth=0.4)

    # colorbar axis: same height as map, with padding
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad="4%")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="x")

    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_annual_soc_line(annual_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(annual_df["year"], annual_df["soc_mean_high_lai"], linewidth=1.6)
    ax.set_title(f"Annual mean SOC where mean LAI > {LAI_THRESHOLD} (1950–2024)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean SOC (same unit as Total_C)")
    ax.grid(True, alpha=0.3)
    plt.savefig(FIG_DIR / "Annual_SOC_highLAI_1950_2024.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# (9) Main
# =============================================================================
def main():
    # Pick a sample SOC parquet to define the SOC grid
    sample_fp = None
    for y in range(START_YEAR, END_YEAR + 1):
        for m in range(1, 13):
            fp = parquet_path(y, m)
            if fp.exists():
                sample_fp = fp
                break
        if sample_fp is not None:
            break
    if sample_fp is None:
        raise FileNotFoundError("Could not find any SOC parquet files to define the grid. Check YEAR_DIRS.")

    base_index, lon_sorted, lat_sorted_desc = build_base_index_from_soc_parquet(sample_fp)

    # 1) Mean LAI map from NetCDF
    mean_lai_df = read_lai_points_mean_1950_2024()
    mean_lai_1d = align_lai_to_soc_grid(mean_lai_df, base_index)
    grid_lai = reshape_to_grid(mean_lai_1d, base_index, lon_sorted, lat_sorted_desc)

    plot_map_imshow(
        grid_lai,
        lon_sorted, lat_sorted_desc,
        title="Mean LAI (1950–2024)",
        cbar_label="LAI",
        out_path=OUT_LAI_MAP,
        cmap="viridis",
        vmin=None, vmax=None
    )

    # High-LAI mask
    high_mask = np.isfinite(mean_lai_1d) & (mean_lai_1d > LAI_THRESHOLD)
    if not np.any(high_mask):
        raise ValueError(f"No cells exceed mean LAI threshold {LAI_THRESHOLD}. Check threshold or LAI alignment.")

    # 2) SOC mean map for high-LAI cells + annual series
    mean_soc_high_1d, annual_df = compute_soc_mean_map_and_annual(base_index, high_mask)
    grid_soc_high = reshape_to_grid(mean_soc_high_1d, base_index, lon_sorted, lat_sorted_desc)

    plot_map_imshow(
        grid_soc_high,
        lon_sorted, lat_sorted_desc,
        title=f"Mean SOC (1950–2024) where mean LAI > {LAI_THRESHOLD}",
        cbar_label="SOC (g/kg)",   # change if your Total_C unit differs
        out_path=OUT_SOC_MAP,
        cmap="viridis",
        vmin=0, vmax=30
    )

    # Save annual CSV and a quick line plot (optional)
    annual_df.to_csv(OUT_SOC_CSV, index=False)
    print(f"Saved annual SOC CSV: {OUT_SOC_CSV}")
    plot_annual_soc_line(annual_df)

    print("Done.")


if __name__ == "__main__":
    main()
