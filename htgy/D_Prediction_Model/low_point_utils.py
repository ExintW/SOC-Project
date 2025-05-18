#!/usr/bin/env python3
# low_point_utils.py

import os
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter

# Append parent directory to path to access 'globals'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # Expects DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.transform import rowcol
from pyproj import Transformer

# ——————————————————————————————————————————————————————————————
# CONFIGURATION
# ——————————————————————————————————————————————————————————————
REGION_CSV = PROCESSED_DIR / "Resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
TIF_PATH   = DATA_DIR       / "htgyDEM.tif"
OUTPUT_CSV = PROCESSED_DIR     / "Low_Point_Summary.csv"

# Pre-create a lon/lat → UTM transformer for UTM zone 49N
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)


# ——————————————————————————————————————————————————————————————
# STEP 1: build a 2D DEM grid at 1 km resolution from your CSV
# ——————————————————————————————————————————————————————————————
def create_grid(df, lon_col, lat_col, val_col):
    pivoted = df.pivot(index=lat_col, columns=lon_col, values=val_col)
    pivoted = pivoted.sort_index(ascending=False)
    return pivoted.values

def load_1km_grid():
    df = pd.read_csv(REGION_CSV, encoding="utf-8-sig")
    lon_col, lat_col = "LON", "LAT"
    grid_x = np.sort(df[lon_col].unique())
    grid_y = np.sort(df[lat_col].unique())[::-1]
    dem    = create_grid(df, lon_col, lat_col, "htgy_DEM")
    return dem, grid_x, grid_y


# ——————————————————————————————————————————————————————————————
# STEP 2: detect 1 km low-points and compute capacity/difference
# ——————————————————————————————————————————————————————————————
def precompute_low_point(dem):
    area = 10 * 10  # m² per 1 km cell
    fp   = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=bool)
    neigh_min = minimum_filter(dem, footprint=fp, mode="nearest")
    low_mask  = neigh_min > dem

    cap = np.zeros_like(dem, dtype=float)
    cap[low_mask] = area * (neigh_min[low_mask] - dem[low_mask])

    dif = np.zeros_like(dem, dtype=float)
    dif[low_mask] = neigh_min[low_mask] - dem[low_mask]
    dif[dif == 0] = np.nan

    return low_mask, cap, dif


# ——————————————————————————————————————————————————————————————
# STEP 3: resample to 10 m, detect 10 m low-points, and count per 1 km cell
# ——————————————————————————————————————————————————————————————
def summarize_low_point_density(low_mask, grid_x, grid_y, tif_path, output_csv):
    with rasterio.open(tif_path) as src:
        if src.crs.is_geographic:
            dst_crs, res = "EPSG:32649", (10, 10)
            dst_t, w, h = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height,
                *src.bounds, resolution=res
            )
            dem_hr = np.empty((h, w), dtype=src.meta["dtype"])
            reproject(
                source=rasterio.band(src, 1),
                destination=dem_hr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_t,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
            transform_hr = dst_t
        else:
            res_x, res_y = src.res
            sx, sy       = res_x/10.0, res_y/10.0
            out_w  = max(1, int(src.width  * sx))
            out_h  = max(1, int(src.height * sy))
            dem_hr = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
            transform_hr = src.transform * src.transform.scale(
                (src.width  / out_w),
                (src.height / out_h)
            )

    # detect 10 m low-pixels
    fp_hr     = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=bool)
    neigh_min = minimum_filter(dem_hr, footprint=fp_hr, mode="nearest")
    low_hr    = neigh_min > dem_hr

    records = []
    for (r, c) in np.argwhere(low_mask):
        lon, lat = grid_x[c], grid_y[r]
        x_utm, y_utm = _TRANSFORMER.transform(lon, lat)
        i0, j0       = rowcol(transform_hr, x_utm, y_utm)
        i0 = np.clip(i0, 0, dem_hr.shape[0]-1)
        j0 = np.clip(j0, 0, dem_hr.shape[1]-1)

        rad = int((1000/10) / 2)  # 1 km / 10 m = 100 px → half-window = 50
        block = low_hr[
            max(0, i0-rad):min(dem_hr.shape[0], i0+rad),
            max(0, j0-rad):min(dem_hr.shape[1], j0+rad)
        ]
        records.append({"LON": lon, "LAT": lat, "low_10m_count": int(block.sum())})

    pd.DataFrame(records)[["LON", "LAT", "low_10m_count"]].to_csv(output_csv, index=False)
    print(f"Low-point summary written to {output_csv}")


# ——————————————————————————————————————————————————————————————
# MAIN
# ——————————————————————————————————————————————————————————————
def main():
    dem, grid_x, grid_y = load_1km_grid()
    low_mask, _, _      = precompute_low_point(dem)
    summarize_low_point_density(low_mask, grid_x, grid_y, TIF_PATH, OUTPUT_CSV)


if __name__ == "__main__":
    main()
