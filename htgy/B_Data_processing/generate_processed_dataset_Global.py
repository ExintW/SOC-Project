#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate 10 km x 10 km grid dataset (EPSG:4326) for multiple basins using:
- Boundary shapefile (any CRS)
- SOC raster (any CRS)
- DEM raster (any CRS)
- Global k1 and k2 rasters (any CRS)

Outputs CSV per basin with columns:
LON, LAT, ORGA, DEM,
SOC_k1_fast_pool (1/day), SOC_k2_slow_pool (1/day),
SOC_k1_fast_pool (1/month), SOC_k2_slow_pool (1/month)

Key fixes:
- Grid is built directly in EPSG:4326 degrees (~10km spacing) to keep a perfect lattice
  so pivot does NOT explode later.
- x_m, y_m are still preserved by projecting points to an auto UTM CRS (for smoothing/filling).
"""

# =============================================================================
# (1) Imports
# =============================================================================
import os
import sys
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr

from pyproj import CRS, Transformer
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


# =============================================================================
# (2) Project globals and paths
# =============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import DATA_DIR, PROCESSED_DIR  # expected in your project

# Global k1 and k2 remain the same
K1_TIF = DATA_DIR / "k1_halfDegree.tif"
K2_TIF = DATA_DIR / "k2_halfDegree.tif"

# Basins to process
BASIN_NAMES = [
    "Amazon_river",
    "Mississippi_river",
    "Shebelle_river",
    "Tagus_Douro_river",
]

# File naming convention
BOUNDARY_DIR = DATA_DIR / "Global" / "Basin_Boundary"
SOC_DIR = DATA_DIR / "Global" / "SOC"
DEM_DIR = DATA_DIR / "Global" / "DEM"

# Output subfolder
OUT_DIR = PROCESSED_DIR / "Global_Basin_10km_Grids"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_ENCODING = "utf-8-sig"

# Grid resolution (meters) -> 10 km
GRID_RES_M = 10000.0

# Smoothing settings for monthly k fields
# Old 1 km used sigma=20 cells, for 10 km you should reduce
SMOOTH_SIGMA_CELLS = 2.0

# SOC conversion rule (kept consistent with your old pipeline)
# Old logic: ORGA = ORGA * 10 * 0.58
APPLY_ORGA_CONVERSION = False
ORGA_FACTOR = 10.0 * 0.58


# =============================================================================
# (3) CRS helpers
# =============================================================================
def ensure_crs(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError(
            f"{layer_name} is missing a CRS. Please ensure the corresponding .prj file is present."
        )
    return gdf


def utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    """
    Select an appropriate UTM CRS based on centroid lon/lat.
    """
    zone = int(np.floor((lon + 180.0) / 6.0) + 1)
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return CRS.from_epsg(epsg)


def to_epsg4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs("EPSG:4326")


# =============================================================================
# (4) Build 10 km grid inside boundary (DIRECTLY IN EPSG:4326 DEGREES)
# =============================================================================
def build_10km_grid_points(boundary_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Create ~10 km spaced grid points inside the basin boundary.
    IMPORTANT: grid is created directly in EPSG:4326 degrees so LON/LAT are a regular lattice.

    Returns:
        DataFrame with columns: LON, LAT, x_m, y_m
    """
    boundary_gdf = ensure_crs(boundary_gdf, "Boundary shapefile")
    boundary_wgs84 = to_epsg4326(boundary_gdf)

    centroid = boundary_wgs84.geometry.unary_union.centroid
    lat0 = float(centroid.y)

    km = GRID_RES_M / 1000.0

    # Degree step approximation
    # 1 deg lat ~ 110.574 km
    grid_res_lat = km / 110.574

    # 1 deg lon ~ 111.320*cos(lat) km
    coslat = np.cos(np.deg2rad(lat0))
    coslat = max(coslat, 1e-6)  # avoid divide-by-zero near poles
    grid_res_lon = km / (111.320 * coslat)

    # Build grid in degrees
    minx, miny, maxx, maxy = boundary_wgs84.total_bounds
    x_coords = np.arange(minx, maxx + grid_res_lon, grid_res_lon)
    y_coords = np.arange(miny, maxy + grid_res_lat, grid_res_lat)

    xx, yy = np.meshgrid(x_coords, y_coords)
    flat_lon = xx.ravel()
    flat_lat = yy.ravel()

    grid_wgs84 = gpd.GeoDataFrame(
        {"LON": flat_lon, "LAT": flat_lat},
        geometry=gpd.points_from_xy(flat_lon, flat_lat),
        crs="EPSG:4326",
    )

    # Clip to basin boundary
    boundary_union = boundary_wgs84.unary_union
    grid_wgs84 = grid_wgs84[grid_wgs84.geometry.within(boundary_union)].reset_index(drop=True)

    # Keep x_m, y_m for fill/smoothing operations (project to UTM)
    utm_crs = utm_crs_from_lonlat(float(centroid.x), float(centroid.y))
    grid_utm = grid_wgs84.to_crs(utm_crs)
    grid_wgs84["x_m"] = grid_utm.geometry.x.values
    grid_wgs84["y_m"] = grid_utm.geometry.y.values

    return pd.DataFrame(grid_wgs84.drop(columns="geometry"))


# =============================================================================
# (5) Raster sampling helpers
# =============================================================================
def _build_transformer(src_crs, dst_crs):
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def sample_raster_nearest(raster_path: Path, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Sample raster at lon/lat points using nearest sampling.
    Handles any raster CRS by transforming point coordinates.
    """
    with rasterio.open(raster_path) as ds:
        raster_crs = ds.crs
        if raster_crs is None:
            raise ValueError(f"Raster is missing CRS: {raster_path}")

        transformer = _build_transformer("EPSG:4326", raster_crs)

        xs, ys = transformer.transform(lons, lats)
        coords = list(zip(xs, ys))

        samples = np.array([v[0] if v is not None and len(v) > 0 else np.nan for v in ds.sample(coords)])
        nodata = ds.nodata

        if nodata is not None:
            samples = np.where(samples == nodata, np.nan, samples)

    return samples.astype(float)


def interpolate_tiff_linear_nearest(path: Path, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Interpolate a raster onto point coordinates using:
    - linear interpolation
    - fill missing with nearest

    Uses rioxarray/xarray interpolation and reprojects raster to EPSG:4326 if needed.
    """
    data_array = rioxarray.open_rasterio(path)

    if data_array.rio.crs is None:
        raise ValueError(f"Raster is missing CRS: {path}")

    # Reproject to EPSG:4326 if needed
    if data_array.rio.crs.to_epsg() != 4326:
        data_array = data_array.rio.reproject("EPSG:4326")

    nodata = data_array.rio.nodata
    data_array = data_array.squeeze("band", drop=True)

    # Rename dims to lon/lat
    if "x" in data_array.dims:
        data_array = data_array.rename({"x": "lon"})
    if "y" in data_array.dims:
        data_array = data_array.rename({"y": "lat"})

    if nodata is not None:
        data_array = data_array.where(data_array != nodata)

    lon_da = xr.DataArray(lons, dims="points")
    lat_da = xr.DataArray(lats, dims="points")

    linear = data_array.interp(lon=lon_da, lat=lat_da, method="linear")
    nearest = data_array.interp(lon=lon_da, lat=lat_da, method="nearest")

    return linear.fillna(nearest).values.astype(float)


def fill_missing_nearest_xy(values: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """
    Fill NaN values using nearest neighbor based on projected meter coordinates.
    """
    v = values.copy()
    missing = ~np.isfinite(v)
    if not missing.any():
        return v

    valid = np.isfinite(v)
    if not valid.any():
        return np.full_like(v, np.nan, dtype=float)

    valid_xy = np.column_stack([x_m[valid], y_m[valid]])
    missing_xy = np.column_stack([x_m[missing], y_m[missing]])

    tree = cKDTree(valid_xy)
    _, idx = tree.query(missing_xy, k=1)
    v[missing] = v[valid][idx]
    return v


# =============================================================================
# (6) k1 and k2 conversions and smoothing
# =============================================================================
def convert_som_to_soc_monthly(k_day: np.ndarray) -> np.ndarray:
    """
    Same conversion used in your original script:
    SOC monthly fraction = (1 - exp(-k_day * 30)) * 0.58
    """
    k_day = np.maximum(k_day, 0.0)
    return (1.0 - np.exp(-k_day * 30.0)) * 0.58


def smooth_monthly_field_on_grid(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Smooth a monthly k field using a 2D gaussian filter on a regular LON/LAT grid.
    IMPORTANT:
    - Do NOT use x_m/y_m as grid axes (UTM x/y are NOT separable from lat/lon).
    - Use unique LON and LAT to form the grid axes.
    """

    work = df[["LON", "LAT", col]].copy()

    # LON ascending, LAT ascending (or descending both okay, just consistent)
    lons = np.sort(work["LON"].unique())
    lats = np.sort(work["LAT"].unique())

    nx = len(lons)
    ny = len(lats)

    lon_to_ix = {v: i for i, v in enumerate(lons)}
    lat_to_iy = {v: i for i, v in enumerate(lats)}

    # Build grid matrix
    mat = np.full((ny, nx), np.nan, dtype=float)

    for _, row in work.iterrows():
        ix = lon_to_ix[row["LON"]]
        iy = lat_to_iy[row["LAT"]]
        mat[iy, ix] = row[col]

    # Fill NaNs by nearest in index space
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    vals = mat.ravel()

    valid = np.isfinite(vals)
    if valid.any():
        filled = griddata(points[valid], vals[valid], points, method="nearest").reshape(ny, nx)
    else:
        filled = np.zeros_like(mat)

    # Smooth
    smoothed = gaussian_filter(filled, sigma=SMOOTH_SIGMA_CELLS)

    # Map back to each row
    out_vals = np.zeros(len(df), dtype=float)
    for i, row in enumerate(work.itertuples(index=False)):
        ix = lon_to_ix[row.LON]
        iy = lat_to_iy[row.LAT]
        out_vals[i] = smoothed[iy, ix]

    return pd.Series(out_vals, index=df.index)



def add_k_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add k1 and k2 in day and month units.
    Uses linear interpolation with nearest fallback.
    Then applies smoothing to monthly fields.
    """
    out = df.copy()
    lons = out["LON"].values
    lats = out["LAT"].values

    k1_day = interpolate_tiff_linear_nearest(K1_TIF, lons, lats)
    k2_day = interpolate_tiff_linear_nearest(K2_TIF, lons, lats)

    k1_day = fill_missing_nearest_xy(k1_day, out["x_m"].values, out["y_m"].values)
    k2_day = fill_missing_nearest_xy(k2_day, out["x_m"].values, out["y_m"].values)

    out["SOC_k1_fast_pool (1/day)"] = k1_day * 0.58
    out["SOC_k2_slow_pool (1/day)"] = k2_day * 0.58
    out["SOC_k1_fast_pool (1/month)"] = convert_som_to_soc_monthly(k1_day)
    out["SOC_k2_slow_pool (1/month)"] = convert_som_to_soc_monthly(k2_day)

    out["SOC_k1_fast_pool (1/month)"] = smooth_monthly_field_on_grid(out, "SOC_k1_fast_pool (1/month)")
    out["SOC_k2_slow_pool (1/month)"] = smooth_monthly_field_on_grid(out, "SOC_k2_slow_pool (1/month)")

    return out


# =============================================================================
# (7) Per-basin pipeline
# =============================================================================
def process_one_basin(basin_name: str) -> Path:
    boundary_shp = BOUNDARY_DIR / f"{basin_name}_basin_boundary.shp"
    dem_tif = DEM_DIR / f"{basin_name}_basin_DEM.tif"
    soc_tif = SOC_DIR / f"{basin_name}_SOC.tif"

    if not boundary_shp.exists():
        raise FileNotFoundError(f"Missing boundary shapefile: {boundary_shp}")
    if not dem_tif.exists():
        raise FileNotFoundError(f"Missing DEM raster: {dem_tif}")
    if not soc_tif.exists():
        raise FileNotFoundError(f"Missing SOC raster: {soc_tif}")

    print(f"\n=== Processing basin: {basin_name} ===")

    # (1) Load boundary
    boundary = gpd.read_file(boundary_shp)
    boundary = ensure_crs(boundary, f"{basin_name} boundary")
    boundary_wgs84 = to_epsg4326(boundary)

    # (2) Build 10 km grid inside boundary
    grid_df = build_10km_grid_points(boundary_wgs84)
    print(f"Grid points inside boundary: {len(grid_df)}")

    # (3) Sample SOC and DEM (nearest)
    soc_vals = sample_raster_nearest(soc_tif, grid_df["LON"].values, grid_df["LAT"].values)
    dem_vals = sample_raster_nearest(dem_tif, grid_df["LON"].values, grid_df["LAT"].values)

    # Fill missing values using nearest in meter coords
    soc_vals = fill_missing_nearest_xy(soc_vals, grid_df["x_m"].values, grid_df["y_m"].values)
    dem_vals = fill_missing_nearest_xy(dem_vals, grid_df["x_m"].values, grid_df["y_m"].values)

    # Apply ORGA conversion
    if APPLY_ORGA_CONVERSION:
        orga = soc_vals * ORGA_FACTOR
    else:
        orga = soc_vals.copy()

    grid_df["ORGA"] = orga
    grid_df["DEM"] = dem_vals

    # (4) Add k1 and k2
    grid_df = add_k_parameters(grid_df)

    # (5) Final output columns
    out_cols = [
        "LON",
        "LAT",
        "ORGA",
        "DEM",
        "SOC_k1_fast_pool (1/day)",
        "SOC_k2_slow_pool (1/day)",
        "SOC_k1_fast_pool (1/month)",
        "SOC_k2_slow_pool (1/month)",
    ]
    out_df = grid_df[out_cols].copy()

    out_path = OUT_DIR / f"{basin_name}_10km_SOC_DEM_k1k2.csv"
    out_df.to_csv(out_path, index=False, encoding=CSV_ENCODING)
    print(f"Saved: {out_path}")
    return out_path


# =============================================================================
# (8) Main
# =============================================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting global basin 10 km grid generation...")
    outputs: List[Path] = []

    for basin in BASIN_NAMES:
        try:
            out_csv = process_one_basin(basin)
            outputs.append(out_csv)
        except Exception as e:
            print(f"[ERROR] Basin {basin} failed: {e}")

    print("\nDone.")
    if outputs:
        print("Outputs generated:")
        for p in outputs:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
