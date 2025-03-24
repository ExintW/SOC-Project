"""
SOC Model with River Basin and Dam Effects
=========================================

This script models soil organic carbon (SOC) over the Loess Plateau, accounting for:
 - Partitioning of SOC into fast & slow pools
 - Erosion & deposition using a RUSLE-based approach
 - Dam capacity (for sediment storage)
 - River routing (removing SOC from the system)
 - Monthly climate forcing (LAI, precipitation)
 - Numba-accelerated flow routing

Debugging Tips (CRS, Bounding Box, etc.):
-----------------------------------------
1) Ensure each shapefile is in the correct CRS before reprojecting:
   - If a shapefile says EPSG:32649 (UTM 49N) but is actually lat/lon, you must fix that mismatch.
   - Use .set_crs(...) if .crs is None or incorrect, then .to_crs(...) to convert.

2) Compare bounding boxes in a common CRS (e.g., EPSG:4326) to confirm they overlap:
   - Example:
       print("Grid extent:", grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
       print("Shapefile extent:", shapefile.total_bounds)
   - If they don’t overlap, your plot might only show a tiny sliver or nothing at all.

3) Plot shapefiles alone to confirm geometry is correct:
   - shapefile.plot(...)

4) If you use ax.imshow(..., extent=[minx, maxx, miny, maxy]), you clamp the plot
   to that bounding box. Any data outside it won't appear.

5) Buffering in a projected CRS:
   - If your resolution is in degrees but you buffer in EPSG:3857 (meters),
     the buffer distance might be too small or large. Double-check units.

6) Delete old "PrecomputedMasks.npz" if you change the shapefiles or buffering steps,
   so the script re-rasterizes and doesn’t use stale data.

Following these steps should help ensure you see the full boundary coverage in your final plots.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import netCDF4 as nc
import geopandas as gpd
from shapely.geometry import Point
from shapely import vectorized  # We use precomputed masks instead of per-point vectorized intersection.
from shapely.prepared import prep
from pathlib import Path
import sys
from numba import njit, prange
import numba
from affine import Affine
import rasterio
from rasterio.features import rasterize

# Append parent directory to path to access 'globals' if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from globals import *  # Expects DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

# =============================================================================
# 0) SETUP: READ THE LOESS PLATEAU BORDER SHAPEFILE & SET DESIRED CRS
# =============================================================================

desired_crs = "EPSG:4326"

# Read the Loess Plateau border shapefile and combine all features into one geometry.
loess_border_path = DATA_DIR / "Loess_Plateau_vector_border.shp"
loess_border = gpd.read_file(loess_border_path)

print("Loess border reported CRS:", loess_border.crs)
print("Loess border total_bounds:", loess_border.total_bounds)

# union_all() merges all features; recommended over unary_union in newer geopandas versions.
loess_border_geom = loess_border.union_all()

# Reproject the Loess Plateau border to the desired CRS (if needed).
loess_border = loess_border.to_crs(desired_crs)
loess_border_geom = loess_border.union_all()

# =============================================================================
# 1) CSV READING & GRID SETUP
# =============================================================================
# Define file paths for the region CSV, dam CSV, and SOC proportion CSV.
region_csv_path = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
dam_csv_path = PROCESSED_DIR / "htgy_Dam_with_matched_points.csv"
proportion_csv_path = DATA_DIR / "Fast_Slow_SOC_Proportion.csv"

# Read the CSV files.
df = pd.read_csv(region_csv_path, encoding='utf-8-sig')
df_dam = pd.read_csv(dam_csv_path, encoding='utf-8-sig')
df_prop = pd.read_csv(proportion_csv_path, encoding='utf-8-sig')

# Ensure numeric values for dam construction year and storage.
df_dam["year"] = pd.to_numeric(df_dam["year"], errors="coerce")
df_dam["total_stor"] = pd.to_numeric(df_dam["total_stor"], errors="coerce")
df_dam["deposition"] = pd.to_numeric(df_dam["deposition"], errors="coerce")
df_dam["capacity_remained"] = df_dam["total_stor"] - df_dam["deposition"]

# ---------------------------------------------------------------------
# Define column names from the region CSV.
# ---------------------------------------------------------------------
lon_col, lat_col = "LON", "LAT"
soc_col = "ORGA"        # Initial SOC concentration (g/kg)
dem_col = "htgy_DEM"    # DEM (elevation)
landuse_col = "LANDUSE" # Land use
region_col = "Region"   # Not used later
slope_col = "SLOPE"     # Slope values
k1_col = "SOC_k1_fast_pool (1/month)"  # Fast pool decay rate
k2_col = "SOC_k2_slow_pool (1/month)"  # Slow pool decay rate

# ---------------------------------------------------------------------
# Helper: create 2D grid from CSV by pivoting lat/lon
# ---------------------------------------------------------------------
def create_grid(data, col_name):
    """
    Creates a 2D grid using pivot (lat descending, lon ascending).
    """
    return (
        data.pivot(index=lat_col, columns=lon_col, values=col_name)
        .sort_index(ascending=False)
        .values
    )

# ---------------------------------------------------------------------
# Extract grid coordinates
# ---------------------------------------------------------------------
grid_x = np.sort(df[lon_col].unique())           # ascending
grid_y = np.sort(df[lat_col].unique())[::-1]     # descending for latitude

# ---------------------------------------------------------------------
# Create 2D arrays from CSV
# ---------------------------------------------------------------------
C = create_grid(df, soc_col)  # SOC concentration (g/kg)
C = np.clip(C, None, 13.8)    # Clip values above 13.8
DEM = create_grid(df, dem_col)
SAND = create_grid(df, "SAND")
SILT = create_grid(df, "SILT")
CLAY = create_grid(df, "CLAY")
LANDUSE = create_grid(df, landuse_col)
REGION = create_grid(df, region_col)
SLOPE = create_grid(df, slope_col)
K_fast = create_grid(df, k1_col)
K_slow = create_grid(df, k2_col)

# Fill missing values in some arrays.
DEM = np.nan_to_num(DEM, nan=np.nanmean(DEM))
SLOPE = np.nan_to_num(SLOPE, nan=np.nanmean(SLOPE))
K_fast = np.nan_to_num(K_fast, nan=np.nanmean(K_fast))
K_slow = np.nan_to_num(K_slow, nan=np.nanmean(K_slow))

# =============================================================================
# 2) PARTITION SOC INTO FAST & SLOW POOLS
# =============================================================================
def allocate_fast_slow_soc(C, LANDUSE, proportion_df):
    """
    Partition total SOC into fast and slow pools using percentages from the CSV
    (e.g., farmland might have 30% fast pool, 70% slow).
    """
    prop_dict = {
        row['Type']: {
            'fast': row['Fast SOC(%)'] / 100,
            'slow': row['Slow SOC(%)'] / 100
        }
        for _, row in proportion_df.iterrows()
    }
    rows, cols = LANDUSE.shape
    C_fast = np.zeros((rows, cols))
    C_slow = np.zeros((rows, cols))
    p_fast_grid = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            land_type = LANDUSE[i, j]
            props = prop_dict.get(land_type, {'fast': 0, 'slow': 1})
            C_fast[i, j] = C[i, j] * props['fast']
            C_slow[i, j] = C[i, j] * props['slow']
            p_fast_grid[i, j] = props['fast']
    return C_fast, C_slow, p_fast_grid

C_fast, C_slow, p_fast_grid = allocate_fast_slow_soc(C, LANDUSE, df_prop)

# =============================================================================
# 3) DAM DATA PROCESSING
# =============================================================================
# Convert dam capacity from 10,000 m³ to tons using a bulk density of 1300 t/m³.
BULK_DENSITY = 1300

def find_nearest_index(array, value):
    """Return index of element in array closest to value."""
    return (np.abs(array - value)).argmin()

# =============================================================================
# 4) RUSLE COMPONENTS (MONTHLY)
# =============================================================================
def calculate_r_factor_monthly(rain_month_mm):
    """Compute R factor from monthly precipitation: R = 6.94 * rain_month_mm."""
    return 6.94 * rain_month_mm

def calculate_ls_factor(slope, slope_length=1000):
    """
    Compute LS factor from slope (degrees).
    This is a simplified formula; in real RUSLE, LS depends on slope length, slope steepness, etc.
    """
    slope_rad = np.deg2rad(slope)
    return ((slope_length / 22.13) ** 0.4) * ((np.sin(slope_rad) / 0.0896) ** 1.3)

def calculate_c_factor(lai):
    """Compute C factor from LAI: C = exp(-1.7 * LAI)."""
    return np.exp(-1.7 * lai)

def calculate_p_factor(landuse):
    """Return P factor based on land use category."""
    p_values = {
        "sloping cropland": 0.4,
        "forestland": 0.5,
        "grassland": 0.5,
        "not used": 0.5,
        "terrace": 0.1,
        "dam field": 0.05
    }
    return p_values.get(str(landuse).lower(), 1.0)

K_factor = np.full_like(C, 0.03)
LS_factor = calculate_ls_factor(SLOPE)
P_factor = np.array([
    [calculate_p_factor(LANDUSE[i, j]) for j in range(LANDUSE.shape[1])]
    for i in range(LANDUSE.shape[0])
])

# =============================================================================
# 5) CONVERT SOIL LOSS TO SOC LOSS (g/kg/month)
# =============================================================================
def convert_soil_loss_to_soc_loss_monthly(E_t_ha_month, ORGA_g_per_kg, bulk_density=1300):
    """
    Convert soil loss (t/ha/month) to SOC loss (g/kg/month).
    1 t/ha = 100 g/m². Then multiply by (SOC_concentration / 1000) * bulk_density.
    """
    E_g_m2_month = E_t_ha_month * 100.0
    soc_loss_g_m2_month = E_g_m2_month * (ORGA_g_per_kg / 1000.0) * bulk_density
    return soc_loss_g_m2_month / bulk_density

# =============================================================================
# 6) RASTERIZE RIVER BASIN BOUNDARIES & MAIN RIVER USING PRECOMPUTED MASKS
# =============================================================================
"""
Important Debugging Notes:
--------------------------
 - If your shapefiles have different CRSes (e.g., EPSG:32649 for small/large boundary vs. EPSG:4326 for river),
   you must reproject them carefully to a common projected CRS before buffering.
 - Print bounding boxes before & after reprojecting to confirm they match the region you expect.
 - If 'mask_file' exists from a previous run, it won't re-rasterize. Delete it if you've changed the shapefiles
   or any buffering logic.
"""
dx = np.mean(np.diff(grid_x))
dy = np.mean(np.diff(grid_y))
resolution = np.mean([dx, dy])  # In degrees if grid_x, grid_y are lat/lon
buffer_distance = resolution    # This is in degrees if you're in EPSG:4326.
                                # For real buffering in meters, reproject to e.g. EPSG:3857 or UTM.

# Create affine transform for rasterization.
minx = grid_x[0] - dx / 2
maxy = grid_y[0] + dy / 2  # grid_y[0] is the max lat
transform = Affine.translation(minx, maxy) * Affine.scale(dx, -dy)
out_shape = (len(grid_y), len(grid_x))

mask_file = OUTPUT_DIR / "PrecomputedMasks.npz"

print("=== DEBUG: Grid bounding box ===")
print(f"Grid longitude range: {grid_x.min():.6f} to {grid_x.max():.6f}")
print(f"Grid latitude range : {grid_y.min():.6f} to {grid_y.max():.6f}")
print("================================\n")

if mask_file.exists():
    print("Loading precomputed masks...")
    masks = np.load(mask_file)
    small_boundary_mask = masks["small_boundary_mask"]
    large_boundary_mask = masks["large_boundary_mask"]
    river_mask = masks["river_mask"]
else:
    print("Precomputing masks...")
    small_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "htgy_River_Basin.shp")
    large_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "94_area.shp")
    river_shp = gpd.read_file(DATA_DIR / "China_River" / "ChinaRiver_main.shp")

    # Filter river shapefile to only include features within the Loess Plateau border, if desired:
    river_shp = river_shp.to_crs(desired_crs)
    river_shp = river_shp[river_shp.intersects(loess_border_geom)]

    print("Small boundary reported CRS:", small_boundary_shp.crs)
    print("Small boundary total_bounds:", small_boundary_shp.total_bounds)
    print("Large boundary reported CRS:", large_boundary_shp.crs)
    print("Large boundary total_bounds:", large_boundary_shp.total_bounds)
    print("River reported CRS:", river_shp.crs)
    print("River total_bounds:", river_shp.total_bounds)

    # Reproject shapefiles to a projected CRS (e.g., EPSG:3857) for buffering
    proj_crs = "EPSG:3857"
    small_boundary_proj = small_boundary_shp.to_crs(proj_crs)
    large_boundary_proj = large_boundary_shp.to_crs(proj_crs)
    river_proj = river_shp.to_crs(proj_crs)

    print("=== DEBUG: Reprojected shapefile bounds (EPSG:3857) ===")
    print("Small boundary bounds:", small_boundary_proj.total_bounds)
    print("Large boundary bounds:", large_boundary_proj.total_bounds)
    print("River bounds:", river_proj.total_bounds)
    print("======================================================\n")

    t0 = time.perf_counter()
    small_boundary_buffered_proj = small_boundary_proj.buffer(buffer_distance)
    large_boundary_buffered_proj = large_boundary_proj.buffer(buffer_distance)
    river_buffered_proj = river_proj.buffer(buffer_distance)
    print(f"Buffering completed in {time.perf_counter() - t0:.2f} seconds.")

    # Reproject buffered geometries back to desired_crs (EPSG:4326) for rasterization
    small_boundary_buffered_gs = gpd.GeoSeries(small_boundary_buffered_proj, crs=proj_crs).to_crs(desired_crs)
    large_boundary_buffered_gs = gpd.GeoSeries(large_boundary_buffered_proj, crs=proj_crs).to_crs(desired_crs)
    river_buffered_gs = gpd.GeoSeries(river_buffered_proj, crs=proj_crs).to_crs(desired_crs)

    print("=== DEBUG: Buffered boundaries (EPSG:4326) bounds ===")
    print("Small boundary final bounds:", small_boundary_buffered_gs.total_bounds)
    print("Large boundary final bounds:", large_boundary_buffered_gs.total_bounds)
    print("River final bounds:", river_buffered_gs.total_bounds)
    print("======================================================\n")

    # Rasterize the buffered geometries
    small_boundary_mask = rasterize(
        [(geom, 1) for geom in small_boundary_buffered_gs.geometry],
        out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True
    ).astype(bool)

    large_boundary_mask = rasterize(
        [(geom, 1) for geom in large_boundary_buffered_gs.geometry],
        out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True
    ).astype(bool)

    river_mask = np.zeros(out_shape, dtype=bool)
    for geom in river_buffered_gs.geometry:
        river_mask |= rasterize(
            [(geom, 1)], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True
        ).astype(bool)

    print("Precomputed river_mask count:", np.count_nonzero(river_mask))
    np.savez(mask_file,
             small_boundary_mask=small_boundary_mask,
             large_boundary_mask=large_boundary_mask,
             river_mask=river_mask)

print("Precomputed masks ready.")

# ---------------------------------------------------------------------
# Helper: Compute outlet mask for a boundary.
# ---------------------------------------------------------------------
def compute_outlet_mask(boundary_mask, DEM):
    """
    Identify the lowest DEM cell within a boundary (the 'outlet').
    Returns a boolean mask with True only at the outlet cell.
    """
    outlet_mask = np.zeros_like(boundary_mask, dtype=bool)
    indices = np.where(boundary_mask)
    if len(indices[0]) > 0:
        min_index = np.argmin(DEM[boundary_mask])
        outlet_i = indices[0][min_index]
        outlet_j = indices[1][min_index]
        outlet_mask[outlet_i, outlet_j] = True
    return outlet_mask

small_outlet_mask = compute_outlet_mask(small_boundary_mask, DEM)
large_outlet_mask = compute_outlet_mask(large_boundary_mask, DEM)

# ---------------------------------------------------------------------
# Debug: Visualize boundary and river masks over the grid
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

# By default, imshow with extent sets axis limits to [minx, maxx, miny, maxy].
# If your shapefile extends beyond that, it won't appear.
# For debugging, you can remove the 'extent' or set bigger limits.
X, Y = np.meshgrid(grid_x, grid_y)
ax.imshow(np.zeros_like(DEM), cmap='gray',
          extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
          origin='upper')  # Make sure origin='upper' matches your array orientation

# Overlays:
ax.contour(X, Y, small_boundary_mask, levels=[0.5], colors='orange', linestyles='--', linewidths=1.5)
ax.contour(X, Y, large_boundary_mask, levels=[0.5], colors='green', linestyles='-', linewidths=1.5)
ax.contour(X, Y, river_mask, levels=[0.5], colors='red', linestyles='-', linewidths=1.5)

ax.set_title("Boundary and River Masks Overlay")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()

# =============================================================================
# PRECOMPUTE SORTED INDICES FOR NUMBA ACCELERATION (TO AVOID IN-NUMBA SORTING)
# =============================================================================
rows, cols = DEM.shape
flat_dem = DEM.flatten()
sorted_flat_indices = np.argsort(flat_dem)[::-1]  # descending order
sorted_indices = np.empty((sorted_flat_indices.shape[0], 2), dtype=np.int64)
sorted_indices[:, 0], sorted_indices[:, 1] = np.unravel_index(sorted_flat_indices, (rows, cols))

# =============================================================================
# 7) ROUTE SOIL AND SOC FROM HIGHER CELLS (WITH BASIN & RIVER EFFECTS)
# =============================================================================
# Using Numba for acceleration with a fractional flow approach.

try:
    atomic_add = numba.atomic.add
except AttributeError:
    def atomic_add(array, index, value):
        array[index[0], index[1]] += value
    print("Warning: numba.atomic.add not available; using non-atomic addition (serial mode).")

@njit(parallel=True)
def distribute_soil_and_soc_with_dams_numba(
        E_tcell, S, DEM, dam_capacity_arr, grid_x, grid_y,
        small_boundary_mask, small_outlet_mask,
        large_boundary_mask, large_outlet_mask,
        river_mask, sorted_indices
):
    """
    Routes soil & SOC downhill, accounting for dams and boundary outflow.
    E_tcell: soil loss from each cell (t/cell/month)
    S: SOC mass eroded from each cell (kg/cell/month)
    dam_capacity_arr: per-cell dam capacity (tons)
    DEM: digital elevation model
    river_mask: if a neighbor is a river, that portion of SOC is 'lost'
    boundary_mask: if crossing from inside to outside the boundary, only allowed at the 'outlet'
    """
    rows, cols = DEM.shape
    inflow_soil = np.zeros((rows, cols), dtype=np.float64)
    inflow_soc = np.zeros((rows, cols), dtype=np.float64)
    D_soil = np.zeros((rows, cols), dtype=np.float64)
    D_soc = np.zeros((rows, cols), dtype=np.float64)
    lost_soc = np.zeros((rows, cols), dtype=np.float64)
    total_cells = sorted_indices.shape[0]

    def local_atomic_add(arr, idx, value):
        arr[idx[0], idx[1]] += value

    for idx in prange(total_cells):
        i = sorted_indices[idx, 0]
        j = sorted_indices[idx, 1]

        # 'dep_soil' and 'dep_soc' are what's inflowed from upstream
        dep_soil = inflow_soil[i, j]
        dep_soc = inflow_soc[i, j]

        # If there's a dam with remaining capacity, deposit soil up to that capacity
        if dam_capacity_arr[i, j] > 0.0:
            cap = dam_capacity_arr[i, j]
            if dep_soil <= cap:
                D_soil[i, j] = dep_soil
                D_soc[i, j] = dep_soc
                dam_capacity_arr[i, j] = cap - dep_soil
                excess_soil = 0.0
                excess_soc = 0.0
            else:
                D_soil[i, j] = cap
                fraction_deposited = (cap / dep_soil) if dep_soil > 0.0 else 0.0
                deposited_soc = dep_soc * fraction_deposited
                if deposited_soc < 0.0:
                    deposited_soc = 0.0
                D_soc[i, j] = deposited_soc
                dam_capacity_arr[i, j] = 0.0
                excess_soil = dep_soil - cap
                excess_soc = dep_soc - deposited_soc
            current_inflow_soil = excess_soil
            current_inflow_soc = excess_soc
        else:
            # No capacity; everything just passes through
            D_soil[i, j] = dep_soil
            D_soc[i, j] = dep_soc
            current_inflow_soil = dep_soil
            current_inflow_soc = dep_soc

        # Source from local erosion
        source_soil = E_tcell[i, j]
        source_soc = S[i, j]

        # We look for neighbors that are strictly lower in DEM
        total_slope = 0.0
        neighbor_count = 0
        neighbor_indices = np.empty((8, 2), dtype=np.int64)
        slope_diffs = np.empty(8, dtype=np.float64)

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = i + di
                nj = j + dj
                if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    continue
                if DEM[ni, nj] >= DEM[i, j]:
                    continue
                # If neighbor is a river cell, we consider that portion 'lost'
                # and do not route further from that neighbor
                if river_mask[ni, nj]:
                    dist = np.hypot(di, dj) + 1e-9
                    slope_diff = (DEM[i, j] - DEM[ni, nj]) / dist
                    if slope_diff < 0.0:
                        slope_diff = 0.0
                    local_atomic_add(lost_soc, (i, j), source_soc * slope_diff)
                    continue
                # Check boundary crossing for small/large basins
                if small_boundary_mask[i, j] != small_boundary_mask[ni, nj]:
                    if not small_outlet_mask[i, j]:
                        continue
                if large_boundary_mask[i, j] != large_boundary_mask[ni, nj]:
                    if not large_outlet_mask[i, j]:
                        continue

                dist = np.hypot(di, dj) + 1e-9
                slope_diff = (DEM[i, j] - DEM[ni, nj]) / dist
                if slope_diff < 0.0:
                    slope_diff = 0.0
                total_slope += slope_diff
                neighbor_indices[neighbor_count, 0] = ni
                neighbor_indices[neighbor_count, 1] = nj
                slope_diffs[neighbor_count] = slope_diff
                neighbor_count += 1

        # Distribute local soil & SOC to lower neighbors by slope fraction
        if total_slope > 0:
            for k in range(neighbor_count):
                ni = neighbor_indices[k, 0]
                nj = neighbor_indices[k, 1]
                fraction = slope_diffs[k] / total_slope
                local_atomic_add(inflow_soil, (ni, nj), source_soil * fraction)
                local_atomic_add(inflow_soc, (ni, nj), source_soc * fraction)

    return D_soil, D_soc, inflow_soil, inflow_soc, lost_soc

# =============================================================================
# 8) VEGETATION INPUT & UPDATED SOC DYNAMIC MODEL
# =============================================================================
def vegetation_input(LAI):
    """
    Compute vegetation input based on LAI using an empirical formula.
    E.g., V = a * LAI + b
    """
    return 0.03006183 * LAI + 0.05812277

def soc_dynamic_model(C_fast, C_slow,
                      soc_loss_g_kg_month, D_soil, D_soc, V,
                      K_fast, K_slow, p_fast_grid, dt, M_soil, lost_soc):
    """
    Update SOC pools (g/kg) for one month.
    - Erosion removes SOC (soc_loss_g_kg_month).
    - Deposition adds SOC (converted from D_soc to g/kg).
    - Vegetation adds new SOC input.
    - Reaction (decay) reduces each pool at rates K_fast, K_slow.
    - Lost SOC (e.g., to rivers) is subtracted.
    """
    # Erosion partitioned into fast & slow
    erosion_fast = -soc_loss_g_kg_month * p_fast_grid
    erosion_slow = -soc_loss_g_kg_month * (1 - p_fast_grid)

    # Deposition: (D_soc * 1000) / M_soil -> convert t -> g, then per kg soil
    deposition_concentration = (D_soc * 1000.0) / M_soil
    deposition_fast = deposition_concentration * p_fast_grid
    deposition_slow = deposition_concentration * (1 - p_fast_grid)

    # Vegetation input
    vegetation_fast = V * p_fast_grid
    vegetation_slow = V * (1 - p_fast_grid)

    # Reaction/decay
    reaction_fast = -K_fast * C_fast
    reaction_slow = -K_slow * C_slow

    # Lost SOC partition
    lost_fast = lost_soc * p_fast_grid
    lost_slow = lost_soc * (1 - p_fast_grid)

    # Update each pool
    C_fast_new = np.maximum(
        C_fast + (erosion_fast + deposition_fast + vegetation_fast + reaction_fast - lost_fast) * dt,
        0
    )
    C_slow_new = np.maximum(
        C_slow + (erosion_slow + deposition_slow + vegetation_slow + reaction_slow - lost_slow) * dt,
        0
    )
    return C_fast_new, C_slow_new

# =============================================================================
# 9) HELPER: REGRID CMIP/ERA5 POINT DATA TO 2D GRID
# =============================================================================
def create_grid_from_points(lon_points, lat_points, values, grid_x, grid_y):
    """
    Regrid 1D point data to a 2D grid by assigning each point to the nearest cell center.
    """
    grid = np.full((len(grid_y), len(grid_x)), np.nan)
    for k in range(len(values)):
        j = (np.abs(grid_x - lon_points[k])).argmin()
        i = (np.abs(grid_y - lat_points[k])).argmin()
        grid[i, j] = values[k]
    return grid

# =============================================================================
# 10) FIGURE OUTPUT SETUP & INITIAL PLOT
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots()
cax = ax.imshow(C_fast + C_slow, cmap="viridis",
                extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                origin='upper')
cbar = fig.colorbar(cax, label="SOC (g/kg)")
ax.set_title("Initial SOC Distribution (t = 0)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
ax.ticklabel_format(style='plain', axis='x')
plt.savefig(os.path.join(OUTPUT_DIR, "SOC_initial.png"))
plt.close(fig)

# =============================================================================
# 11) MAIN SIMULATION LOOP (MONTHLY, 2007-2025)
# =============================================================================
CELL_AREA_HA = 100.0  # 1 km² = 100 ha
start_year = 2007
end_year = 2025
global_timestep = 0
M_soil = 1.0e8  # total soil mass per cell (kg)

# Initialize current SOC pools
C_fast_current = C_fast.copy()
C_slow_current = C_slow.copy()

os.makedirs(OUTPUT_DIR / "Figure", exist_ok=True)
os.makedirs(OUTPUT_DIR / "Data", exist_ok=True)

t_sim_start = time.perf_counter()

# Precompute sorted indices for Numba function (descending DEM)
rows, cols = DEM.shape
flat_dem = DEM.flatten()
sorted_flat_indices = np.argsort(flat_dem)[::-1]
sorted_indices = np.empty((sorted_flat_indices.shape[0], 2), dtype=np.int64)
sorted_indices[:, 0], sorted_indices[:, 1] = np.unravel_index(sorted_flat_indices, (rows, cols))

for year in range(start_year, end_year + 1):
    # Filter dams built on or before current year
    df_dam_active = df_dam[df_dam["year"] <= year].copy()
    dam_capacity_arr = np.zeros(DEM.shape, dtype=np.float64)
    for _, row in df_dam_active.iterrows():
        i_idx = find_nearest_index(grid_y, row["y"])
        j_idx = find_nearest_index(grid_x, row["x"])
        capacity_10000_m3 = row["capacity_remained"]
        capacity_tons = capacity_10000_m3 * 10000 * BULK_DENSITY
        dam_capacity_arr[i_idx, j_idx] = capacity_tons

    # Load monthly climate data (NetCDF)
    nc_file = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{year}.nc"
    if not os.path.exists(nc_file):
        print(f"NetCDF file not found for year {year}: {nc_file}")
        continue

    with nc.Dataset(nc_file) as ds:
        valid_time = ds.variables['valid_time'][:]  # Expect 12 months
        n_time = len(valid_time)
        lon_nc = ds.variables['longitude'][:]
        lat_nc = ds.variables['latitude'][:]
        lai_data = ds.variables['lai_lv'][:]  # shape: (12, n_points)
        tp_data = ds.variables['tp'][:]       # shape: (12, n_points), in meters

        for month_idx in range(n_time):
            # Regrid LAI data
            lai_1d = lai_data[month_idx, :]
            LAI_2D = create_grid_from_points(lon_nc, lat_nc, lai_1d, grid_x, grid_y)
            LAI_2D = np.nan_to_num(LAI_2D, nan=np.nanmean(LAI_2D))

            # Regrid precipitation and convert to mm
            tp_1d = tp_data[month_idx, :]
            tp_1d_mm = tp_1d * 1000.0
            RAIN_2D = create_grid_from_points(lon_nc, lat_nc, tp_1d_mm, grid_x, grid_y)
            RAIN_2D = np.nan_to_num(RAIN_2D, nan=np.nanmean(RAIN_2D))

            # Compute RUSLE factors
            R_month = calculate_r_factor_monthly(RAIN_2D)
            C_factor_2D = calculate_c_factor(LAI_2D)

            # Calculate soil loss (t/ha/month) & then per cell
            E_t_ha_month = R_month * K_factor * LS_factor * C_factor_2D * P_factor
            E_tcell_month = E_t_ha_month * CELL_AREA_HA

            # Compute SOC mass eroded (kg/cell/month)
            S = E_tcell_month * (C_fast_current + C_slow_current)
            SOC_loss_g_kg_month = convert_soil_loss_to_soc_loss_monthly(
                E_t_ha_month, (C_fast_current + C_slow_current)
            )

            # Call the Numba-accelerated routing function
            D_soil, D_soc, inflow_soil, inflow_soc, lost_soc = distribute_soil_and_soc_with_dams_numba(
                E_tcell_month, S, DEM, dam_capacity_arr, grid_x, grid_y,
                small_boundary_mask, compute_outlet_mask(small_boundary_mask, DEM),
                large_boundary_mask, compute_outlet_mask(large_boundary_mask, DEM),
                river_mask, sorted_indices
            )

            # Debug: Print lost SOC summary
            mean_lost = np.mean(lost_soc)
            max_lost = np.max(lost_soc)
            min_lost = np.min(lost_soc)
            print(f"Year {year} Month {month_idx+1}: Lost_SOC - mean: {mean_lost:.2f}, "
                  f"max: {max_lost:.2f}, min: {min_lost:.2f}")

            # Compute vegetation input
            V = vegetation_input(LAI_2D)

            # Update SOC pools
            C_fast_current, C_slow_current = soc_dynamic_model(
                C_fast_current, C_slow_current,
                SOC_loss_g_kg_month, D_soil, D_soc, V,
                K_fast, K_slow, p_fast_grid,
                dt=1,
                M_soil=M_soil,
                lost_soc=lost_soc
            )

            global_timestep += 1
            print(f"Completed simulation for Year {year}, Month {month_idx+1}")

            # Save figure output
            fig, ax = plt.subplots()
            cax = ax.imshow(C_fast_current + C_slow_current, cmap="viridis",
                            extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                            origin='upper')
            cbar = fig.colorbar(cax, label="SOC (g/kg)")
            ax.set_title(f"SOC at Timestep {global_timestep} (Year {year}, Month {month_idx+1})")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='x')
            filename_fig = f"SOC_{year}_{month_idx+1:02d}_timestep_{global_timestep}_River.png"
            plt.savefig(os.path.join(OUTPUT_DIR, "Figure", filename_fig))
            plt.close(fig)

            # Save CSV output with all terms including lost SOC
            rows_grid, cols_grid = C_fast_current.shape
            lat_list = []
            lon_list = []
            landuse_list = []
            C_fast_list = []
            C_slow_list = []
            # Erosion, deposition, vegetation, reaction
            erosion_fast_list = []
            erosion_slow_list = []
            deposition_fast_list = []
            deposition_slow_list = []
            vegetation_fast_list = []
            vegetation_slow_list = []
            reaction_fast_list = []
            reaction_slow_list = []
            E_t_ha_list = []
            lost_soc_list = []

            # Gather data for CSV
            for i in range(rows_grid):
                for j in range(cols_grid):
                    cell_lon = grid_x[j]
                    cell_lat = grid_y[i]
                    lat_list.append(cell_lat)
                    lon_list.append(cell_lon)
                    landuse_list.append(str(LANDUSE[i, j]))
                    C_fast_list.append(C_fast_current[i, j])
                    C_slow_list.append(C_slow_current[i, j])

                    # Erosion
                    erosion_fast_list.append(-SOC_loss_g_kg_month[i, j] * p_fast_grid[i, j])
                    erosion_slow_list.append(-SOC_loss_g_kg_month[i, j] * (1 - p_fast_grid[i, j]))

                    # Deposition
                    deposition_concentration = (D_soc[i, j] * 1000.0) / M_soil
                    deposition_fast_list.append(deposition_concentration * p_fast_grid[i, j])
                    deposition_slow_list.append(deposition_concentration * (1 - p_fast_grid[i, j]))

                    # Vegetation
                    vegetation_fast_list.append(V[i, j] * p_fast_grid[i, j])
                    vegetation_slow_list.append(V[i, j] * (1 - p_fast_grid[i, j]))

                    # Reaction
                    reaction_fast_list.append(-K_fast[i, j] * C_fast_current[i, j])
                    reaction_slow_list.append(-K_slow[i, j] * C_slow_current[i, j])

                    E_t_ha_list.append(E_t_ha_month[i, j])
                    lost_soc_list.append(lost_soc[i, j])

            df_out = pd.DataFrame({
                'LAT': lat_list,
                'LON': lon_list,
                'Landuse': landuse_list,
                'C_fast': C_fast_list,
                'C_slow': C_slow_list,
                'Erosion_fast': erosion_fast_list,
                'Erosion_slow': erosion_slow_list,
                'Deposition_fast': deposition_fast_list,
                'Deposition_slow': deposition_slow_list,
                'Vegetation_fast': vegetation_fast_list,
                'Vegetation_slow': vegetation_slow_list,
                'Reaction_fast': reaction_fast_list,
                'Reaction_slow': reaction_slow_list,
                'E_t_ha_month': E_t_ha_list,
                'Lost_SOC_River': lost_soc_list
            })

            filename_csv = f"SOC_terms_{year}_{month_idx+1:02d}_timestep_{global_timestep}_River.csv"
            df_out.to_csv(os.path.join(OUTPUT_DIR, "Data", filename_csv), index=False)
            print(f"Saved CSV output for Year {year}, Month {month_idx+1} as {filename_csv}")

print(f"Simulation complete. Total simulation time: {time.perf_counter() - t_sim_start:.2f} seconds.")
print("Final SOC distribution is in C_fast_current + C_slow_current.")
