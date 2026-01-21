import os
import sys
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

# ============================================================
# 0) Project imports
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # DATA_DIR, PROCESSED_DIR


# ============================================================
# 1) Paths
# ============================================================
SOC_TIF_DIR = DATA_DIR / "Global" / "SOC"
GRID_DIR = PROCESSED_DIR / "Global_Basin_10km_Grids"

OUT_DIR = PROCESSED_DIR / "SOC_2001_Basin_NPZ_10km"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2) Basin file mapping (you only edit names here if needed)
# ============================================================
BASINS = {
    "Amazon_river": {
        "tif": SOC_TIF_DIR / "Amazon_river_SOC.tif",
        "grid_stem": "Amazon_river_10km_SOC_DEM_k1k2",
    },
    "Mississippi_river": {
        "tif": SOC_TIF_DIR / "Mississippi_river_SOC.tif",
        "grid_stem": "Mississippi_river_10km_SOC_DEM_k1k2",
    },
    "Shebelle_river": {
        "tif": SOC_TIF_DIR / "Shebelle_river_SOC.tif",
        "grid_stem": "Shebelle_river_10km_SOC_DEM_k1k2",
    },
    "Tagus_Douro_river": {
        "tif": SOC_TIF_DIR / "Tagus_Douro_river_SOC.tif",
        "grid_stem": "Tagus_Douro_river_10km_SOC_DEM_k1k2",
    },
}


# ============================================================
# 3) Helpers
# ============================================================
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of these columns: {candidates}")

def find_grid_file_by_stem(folder: Path, stem: str) -> Path:
    """
    Finds the grid file with this stem in GRID_DIR.
    Supports .csv, .xlsx, .xls.
    """
    candidates = []
    for ext in ["*.csv", "*.xlsx", "*.xls"]:
        candidates.extend(folder.glob(stem + ext))

    if len(candidates) == 0:
        raise FileNotFoundError(f"Cannot find grid file for stem: {stem} in {folder}")
    if len(candidates) > 1:
        # choose the most recently modified
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)

    return candidates[0]

def read_grid_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def sample_soc_from_tif_to_points(tif_path: Path, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        raster_crs = src.crs

        # Convert lon, lat into raster CRS if needed
        if raster_crs is not None and not raster_crs.is_geographic:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            xs, ys = transformer.transform(lons, lats)
            coords = list(zip(xs, ys))
        else:
            coords = list(zip(lons, lats))

        sampled = np.array([v[0] for v in src.sample(coords)], dtype="float32")

    if nodata is not None:
        sampled[sampled == nodata] = np.nan

    return sampled


# ============================================================
# 4) Run all 4 basins
# ============================================================
for basin_name, info in BASINS.items():
    tif_path = info["tif"]
    grid_stem = info["grid_stem"]

    print("\n============================================================")
    print(f"Basin: {basin_name}")
    print("SOC tif:", tif_path)
    print("Grid stem:", grid_stem)
    print("============================================================")

    if not tif_path.exists():
        print(f"⚠️ Missing tif file: {tif_path}")
        continue

    # Find the basin grid file (csv or excel)
    grid_path = find_grid_file_by_stem(GRID_DIR, grid_stem)
    print("Grid file:", grid_path)

    # Read grid
    grid = read_grid_file(grid_path)

    # Detect lon and lat columns
    LON_COL = find_col(grid, ["LON", "lon", "Longitude", "longitude"])
    LAT_COL = find_col(grid, ["LAT", "lat", "Latitude", "latitude"])

    lons = grid[LON_COL].values.astype(float)
    lats = grid[LAT_COL].values.astype(float)

    # Sample SOC at grid points
    soc_2001 = sample_soc_from_tif_to_points(tif_path, lons, lats)
    grid["soc_2001"] = soc_2001

    # Print stats
    valid_n = np.isfinite(soc_2001).sum()
    mean_soc = np.nanmean(soc_2001)
    print(f"Valid points: {valid_n} / {len(grid)}")
    print(f"Mean SOC 2001 (valid only): {mean_soc:.4f} g/kg")

    # Pivot to 2D matrix
    pivot = grid.pivot(index=LAT_COL, columns=LON_COL, values="soc_2001")
    lat_vals = pivot.index.values
    lon_vals = pivot.columns.values
    soc_mat = pivot.values

    # Save NPZ
    out_npz = OUT_DIR / f"soc_2001_{basin_name}_10km_matrix.npz"
    np.savez(
        out_npz,
        lon=lon_vals,
        lat=lat_vals,
        soc_mean_matrix=soc_mat,
        basin=basin_name
    )

    print(f"✅ Saved NPZ: {out_npz}")
    print(f"Matrix shape: {soc_mat.shape}")

print("\nAll basins finished ✅")
