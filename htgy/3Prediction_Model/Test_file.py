"""
SOC Model with River Basin and Dam Effects
=========================================

此脚本在原有基础上：
 - 读取 CSV 网格 & Shapefile (River Basin, 94_area, ChinaRiver_main)
 - 做正确投影 (EPSG:4326) + 米制缓冲 (EPSG:3857) + 再投回EPSG:4326
 - 生成并栅格化 small/large boundary、river 掩膜
 - 进行 RUSLE + SOC 动力学模拟
 - **改进：可视化时用初始 SOC/DEM 作为背景，避免全黑画面，便于调试**
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
from shapely import vectorized
from shapely.prepared import prep
from pathlib import Path
import sys
from numba import njit, prange
import numba
from affine import Affine
import rasterio
from rasterio.features import rasterize

# 如果 globals.py 放在父目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from htgy.globals import DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

# =============================================================================
# 0) SETUP: READ THE LOESS PLATEAU BORDER & DESIRED CRS
# =============================================================================

desired_crs = "EPSG:4326"  # 网格也是EPSG:4326

loess_border_path = DATA_DIR / "Loess_Plateau_vector_border.shp"
loess_border = gpd.read_file(loess_border_path)
print("Loess Plateau border - original CRS:", loess_border.crs)
print("Loess Plateau total_bounds:", loess_border.total_bounds)

# 兼容新版本: union_all() 取代 unary_union
# loess_border_geom = loess_border.unary_union
loess_border_geom = loess_border.union_all()

loess_border = loess_border.to_crs(desired_crs)
loess_border_geom = loess_border.union_all()

# =============================================================================
# 1) CSV READING & GRID SETUP
# =============================================================================

region_csv_path = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
dam_csv_path = PROCESSED_DIR / "htgy_Dam_with_matched_points.csv"
proportion_csv_path = DATA_DIR / "Fast_Slow_SOC_Proportion.csv"

df = pd.read_csv(region_csv_path, encoding='utf-8-sig')
df_dam = pd.read_csv(dam_csv_path, encoding='utf-8-sig')
df_prop = pd.read_csv(proportion_csv_path, encoding='utf-8-sig')

df_dam["year"] = pd.to_numeric(df_dam["year"], errors="coerce")
df_dam["total_stor"] = pd.to_numeric(df_dam["total_stor"], errors="coerce")
df_dam["deposition"] = pd.to_numeric(df_dam["deposition"], errors="coerce")
df_dam["capacity_remained"] = df_dam["total_stor"] - df_dam["deposition"]

lon_col, lat_col = "LON", "LAT"
soc_col = "ORGA"
dem_col = "htgy_DEM"
landuse_col = "LANDUSE"
region_col = "Region"
slope_col = "SLOPE"
k1_col = "SOC_k1_fast_pool (1/month)"
k2_col = "SOC_k2_slow_pool (1/month)"

def create_grid(data, col_name):
    return (
        data.pivot(index=lat_col, columns=lon_col, values=col_name)
        .sort_index(ascending=False)
        .values
    )

grid_x = np.sort(df[lon_col].unique())
grid_y = np.sort(df[lat_col].unique())[::-1]

C = create_grid(df, soc_col)
C = np.clip(C, None, 13.8)
DEM = create_grid(df, dem_col)
SAND = create_grid(df, "SAND")
SILT = create_grid(df, "SILT")
CLAY = create_grid(df, "CLAY")
LANDUSE = create_grid(df, landuse_col)
REGION = create_grid(df, region_col)
SLOPE = create_grid(df, slope_col)
K_fast = create_grid(df, k1_col)
K_slow = create_grid(df, k2_col)

DEM = np.nan_to_num(DEM, nan=np.nanmean(DEM))
SLOPE = np.nan_to_num(SLOPE, nan=np.nanmean(SLOPE))
K_fast = np.nan_to_num(K_fast, nan=np.nanmean(K_fast))
K_slow = np.nan_to_num(K_slow, nan=np.nanmean(K_slow))

print("\n=== [Debug] CSV Grid Extent ===")
print("Longitude range:", grid_x.min(), grid_x.max())
print("Latitude  range:", grid_y.min(), grid_y.max())

# =============================================================================
# 2) PARTITION SOC INTO FAST & SLOW POOLS
# =============================================================================
def allocate_fast_slow_soc(C, LANDUSE, proportion_df):
    prop_dict = {
        row['Type']: {
            'fast': row['Fast SOC(%)'] / 100,
            'slow': row['Slow SOC(%)'] / 100
        }
        for _, row in proportion_df.iterrows()
    }
    rows, cols = LANDUSE.shape
    C_fast_ = np.zeros((rows, cols))
    C_slow_ = np.zeros((rows, cols))
    p_fast_grid_ = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            land_type = LANDUSE[i, j]
            props = prop_dict.get(land_type, {'fast': 0, 'slow': 1})
            C_fast_[i, j] = C[i, j] * props['fast']
            C_slow_[i, j] = C[i, j] * props['slow']
            p_fast_grid_[i, j] = props['fast']
    return C_fast_, C_slow_, p_fast_grid_

C_fast, C_slow, p_fast_grid = allocate_fast_slow_soc(C, LANDUSE, df_prop)

# =============================================================================
# 3) DAM DATA PROCESSING
# =============================================================================
BULK_DENSITY = 1300
def find_nearest_index(array, value):
    return (np.abs(array - value)).argmin()

# =============================================================================
# 4) RUSLE COMPONENTS (MONTHLY)
# =============================================================================

def calculate_r_factor_monthly(rain_month_mm):
    return 6.94 * rain_month_mm

def calculate_ls_factor(slope, slope_length=1000):
    slope_rad = np.deg2rad(slope)
    return ((slope_length / 22.13)**0.4)*((np.sin(slope_rad)/0.0896)**1.3)

def calculate_c_factor(lai):
    return np.exp(-1.7 * lai)

def calculate_p_factor(landuse):
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
    E_g_m2_month = E_t_ha_month*100.0
    soc_loss_g_m2_month = E_g_m2_month*(ORGA_g_per_kg/1000.0)*bulk_density
    return soc_loss_g_m2_month / bulk_density

# =============================================================================
# 6) RASTERIZE RIVER BASIN & MAIN RIVER (CORRECT PROJECTION/BUFFER)
# =============================================================================

dx = np.mean(np.diff(grid_x))
dy = np.mean(np.diff(grid_y))

minx = grid_x[0] - dx/2
maxy = grid_y[0] + dy/2
transform = Affine.translation(minx, maxy) * Affine.scale(dx, -dy)
out_shape = (len(grid_y), len(grid_x))

mask_file = OUTPUT_DIR / "PrecomputedMasks.npz"

print("\n=== [Debug] Grid bounding box (EPSG:4326) ===")
print("Longitude:", grid_x.min(), grid_x.max())
print("Latitude: ", grid_y.min(), grid_y.max())
print("=============================================\n")

if mask_file.exists():
    print("[Info] Found mask file, loading from:", mask_file)
    masks = np.load(mask_file)
    small_boundary_mask = masks["small_boundary_mask"]
    large_boundary_mask = masks["large_boundary_mask"]
    river_mask = masks["river_mask"]
else:
    print("[Info] No mask_file found (or manually removed). Start computing new masks...")

    small_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "htgy_River_Basin.shp")
    large_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "94_area.shp")
    river_shp = gpd.read_file(DATA_DIR / "China_River" / "ChinaRiver_main.shp")

    print("=== [Debug] ORIGINAL Shapefiles ===")
    print("Small_Basin:", small_boundary_shp.crs, small_boundary_shp.total_bounds)
    print("94_area:", large_boundary_shp.crs, large_boundary_shp.total_bounds)
    print("ChinaRiver_main:", river_shp.crs, river_shp.total_bounds)

    # 1) 先转EPSG:4326
    small_boundary_4326 = small_boundary_shp.to_crs(desired_crs)
    large_boundary_4326 = large_boundary_shp.to_crs(desired_crs)
    river_4326 = river_shp.to_crs(desired_crs)

    print("\n=== [Debug] After reproject to EPSG:4326 ===")
    print("Small boundary:", small_boundary_4326.crs, small_boundary_4326.total_bounds)
    print("94_area:", large_boundary_4326.crs, large_boundary_4326.total_bounds)
    print("ChinaRiver:", river_4326.crs, river_4326.total_bounds)

    # 2) 缓冲1km => EPSG:3857 -> buffer(1000) -> 回EPSG:4326
    buffer_distance_m = 1000.0
    proj_crs = "EPSG:3857"

    small_basin_proj = small_boundary_4326.to_crs(proj_crs)
    large_area_proj = large_boundary_4326.to_crs(proj_crs)
    river_proj = river_4326.to_crs(proj_crs)

    print("\n=== [Debug] In EPSG:3857 for buffer ===")
    print("Small_Basin:", small_basin_proj.total_bounds)
    print("94_area:", large_area_proj.total_bounds)
    print("ChinaRiver:", river_proj.total_bounds)

    t0 = time.perf_counter()
    small_buffered_proj = small_basin_proj.buffer(buffer_distance_m)
    large_buffered_proj = large_area_proj.buffer(buffer_distance_m)
    river_buffered_proj = river_proj.buffer(buffer_distance_m)
    print(f"[Info] Buffering took {time.perf_counter()-t0:.2f} s")

    small_buffered_4326 = gpd.GeoSeries(small_buffered_proj, crs=proj_crs).to_crs(desired_crs)
    large_buffered_4326 = gpd.GeoSeries(large_buffered_proj, crs=proj_crs).to_crs(desired_crs)
    river_buffered_4326 = gpd.GeoSeries(river_buffered_proj, crs=proj_crs).to_crs(desired_crs)

    print("\n=== [Debug] After final reproject to EPSG:4326 ===")
    print("Small boundary final bounds:", small_buffered_4326.total_bounds)
    print("94_area final bounds:", large_buffered_4326.total_bounds)
    print("ChinaRiver final bounds:", river_buffered_4326.total_bounds)

    # 3) 栅格化
    small_boundary_mask = rasterize(
        [(geom, 1) for geom in small_buffered_4326.geometry if geom is not None],
        out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True
    ).astype(bool)

    large_boundary_mask = rasterize(
        [(geom, 1) for geom in large_buffered_4326.geometry if geom is not None],
        out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True
    ).astype(bool)

    river_mask = np.zeros(out_shape, dtype=bool)
    for geom in river_buffered_4326.geometry:
        if geom is None or geom.is_empty:
            continue
        mask_ = rasterize(
            [(geom, 1)], out_shape=out_shape, transform=transform,
            fill=0, dtype=np.uint8, all_touched=True
        ).astype(bool)
        river_mask |= mask_

    print("Final small_boundary_mask sum:", np.count_nonzero(small_boundary_mask))
    print("Final large_boundary_mask sum:", np.count_nonzero(large_boundary_mask))
    print("Final river_mask sum:", np.count_nonzero(river_mask))

    np.savez(mask_file,
             small_boundary_mask=small_boundary_mask,
             large_boundary_mask=large_boundary_mask,
             river_mask=river_mask)
    print("[Info] Saved new masks =>", mask_file)

print("[Info] Raster mask generation done.\n")

def compute_outlet_mask(boundary_mask, DEM):
    outlet_mask = np.zeros_like(boundary_mask, dtype=bool)
    indices = np.where(boundary_mask)
    if len(indices[0])>0:
        min_index = np.argmin(DEM[boundary_mask])
        outlet_i = indices[0][min_index]
        outlet_j = indices[1][min_index]
        outlet_mask[outlet_i, outlet_j] = True
    return outlet_mask

small_outlet_mask = compute_outlet_mask(small_boundary_mask, DEM)
large_outlet_mask = compute_outlet_mask(large_boundary_mask, DEM)

# =============================================================================
# 可视化debug (用初始SOC C 或 DEM做背景, 避免纯黑)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# 构建坐标网格
X, Y = np.meshgrid(grid_x, grid_y)

# 这里使用初始SOC (C_fast + C_slow) 作为背景
# 若想看DEM, 可改为 background_data = DEM
background_data = C_fast + C_slow
cmap_bg = "gray_r"  # 您可换成 "viridis" / "terrain" / "Spectral" 等

# 显示背景SOC
im = ax.imshow(background_data, cmap=cmap_bg,
               extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
               origin='upper', alpha=0.6)

cbar = fig.colorbar(im, ax=ax, label="Initial SOC (g/kg)")

# 叠加 contour 显示 small_boundary / large_boundary / river
ax.contour(X, Y, small_boundary_mask, levels=[0.5],
           colors='orange', linestyles='--', linewidths=1.5)
ax.contour(X, Y, large_boundary_mask, levels=[0.5],
           colors='green', linestyles='-', linewidths=1.5)
ax.contour(X, Y, river_mask, levels=[0.5],
           colors='red', linestyles='-', linewidths=1.5)

ax.set_title("Boundary & River Masks Overlay (Initial SOC as Background)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
ax.ticklabel_format(style='plain', axis='x')
plt.show()

# =============================================================================
# PRECOMPUTE SORTED INDICES FOR NUMBA
# =============================================================================
rows, cols = DEM.shape
flat_dem = DEM.flatten()
sorted_flat_indices = np.argsort(flat_dem)[::-1]
sorted_indices = np.empty((sorted_flat_indices.shape[0], 2), dtype=np.int64)
sorted_indices[:, 0], sorted_indices[:, 1] = np.unravel_index(sorted_flat_indices, (rows, cols))

# =============================================================================
# 7) ROUTE SOIL AND SOC (WITH DAMS & RIVER)
# =============================================================================

try:
    atomic_add = numba.atomic.add
except AttributeError:
    def atomic_add(arr, idx, val):
        arr[idx[0], idx[1]] += val
    print("Warning: fallback to local add (non-atomic).")

@njit(parallel=True)
def distribute_soil_and_soc_with_dams_numba(
    E_tcell, S, DEM, dam_capacity_arr, grid_x, grid_y,
    small_boundary_mask, small_outlet_mask,
    large_boundary_mask, large_outlet_mask,
    river_mask, sorted_indices
):
    rows, cols = DEM.shape
    inflow_soil = np.zeros((rows, cols), dtype=np.float64)
    inflow_soc = np.zeros((rows, cols), dtype=np.float64)
    D_soil = np.zeros((rows, cols), dtype=np.float64)
    D_soc = np.zeros((rows, cols), dtype=np.float64)
    lost_soc = np.zeros((rows, cols), dtype=np.float64)

    total_cells = sorted_indices.shape[0]

    def local_atomic_add(a, xy, v):
        a[xy[0], xy[1]] += v

    for idx in prange(total_cells):
        i = sorted_indices[idx, 0]
        j = sorted_indices[idx, 1]

        dep_soil = inflow_soil[i,j]
        dep_soc  = inflow_soc[i,j]

        # dam capacity
        if dam_capacity_arr[i,j] > 0.0:
            cap = dam_capacity_arr[i,j]
            if dep_soil <= cap:
                D_soil[i,j] = dep_soil
                D_soc[i,j] = dep_soc
                dam_capacity_arr[i,j] = cap - dep_soil
                excess_soil = 0.0
                excess_soc  = 0.0
            else:
                D_soil[i,j] = cap
                frac_dep = (cap/dep_soil) if dep_soil>0 else 0.0
                dep_soc_ = dep_soc*frac_dep
                if dep_soc_<0: dep_soc_ = 0.0
                D_soc[i,j] = dep_soc_
                dam_capacity_arr[i,j] = 0.0
                excess_soil = dep_soil - cap
                excess_soc  = dep_soc - dep_soc_
            current_inflow_soil = excess_soil
            current_inflow_soc  = excess_soc
        else:
            D_soil[i,j] = dep_soil
            D_soc[i,j]  = dep_soc
            current_inflow_soil = dep_soil
            current_inflow_soc  = dep_soc

        # local erosion
        source_soil = E_tcell[i,j]
        source_soc  = S[i,j]

        total_slope = 0.0
        neighbor_count = 0
        neighbor_ix = np.empty((8, 2), dtype=np.int64)
        slope_diffs  = np.empty(8, dtype=np.float64)

        for di in (-1,0,1):
            for dj in (-1,0,1):
                if di==0 and dj==0:
                    continue
                ni = i+di
                nj = j+dj
                if ni<0 or ni>=rows or nj<0 or nj>=cols:
                    continue
                if DEM[ni,nj]>=DEM[i,j]:
                    continue
                # if neighbor is river
                if river_mask[ni,nj]:
                    dist = np.hypot(di,dj)+1e-9
                    slope_diff = (DEM[i,j]-DEM[ni,nj])/dist
                    if slope_diff<0: slope_diff=0
                    local_atomic_add(lost_soc, (i,j), source_soc*slope_diff)
                    continue
                # boundary crossing
                if small_boundary_mask[i,j]!=small_boundary_mask[ni,nj]:
                    if not small_outlet_mask[i,j]:
                        continue
                if large_boundary_mask[i,j]!=large_boundary_mask[ni,nj]:
                    if not large_outlet_mask[i,j]:
                        continue

                dist = np.hypot(di,dj)+1e-9
                slope_diff = (DEM[i,j]-DEM[ni,nj])/dist
                if slope_diff<0:
                    slope_diff=0
                total_slope += slope_diff
                neighbor_ix[neighbor_count,0] = ni
                neighbor_ix[neighbor_count,1] = nj
                slope_diffs[neighbor_count] = slope_diff
                neighbor_count+=1

        if total_slope>0:
            for k in range(neighbor_count):
                ni = neighbor_ix[k,0]
                nj = neighbor_ix[k,1]
                frac = slope_diffs[k]/total_slope
                local_atomic_add(inflow_soil, (ni,nj), source_soil*frac)
                local_atomic_add(inflow_soc, (ni,nj), source_soc*frac)

    return D_soil, D_soc, inflow_soil, inflow_soc, lost_soc


# =============================================================================
# 8) VEGETATION INPUT & SOC DYNAMIC MODEL
# =============================================================================

def vegetation_input(LAI):
    return 0.03006183*LAI + 0.05812277

def soc_dynamic_model(C_fast, C_slow,
                      soc_loss_g_kg_month, D_soil, D_soc, V,
                      K_fast, K_slow, p_fast_grid,
                      dt, M_soil, lost_soc):
    erosion_fast = -soc_loss_g_kg_month*p_fast_grid
    erosion_slow = -soc_loss_g_kg_month*(1-p_fast_grid)

    dep_conc = (D_soc*1000.0)/M_soil
    deposition_fast = dep_conc*p_fast_grid
    deposition_slow = dep_conc*(1-p_fast_grid)

    vegetation_fast = V*p_fast_grid
    vegetation_slow = V*(1-p_fast_grid)

    reaction_fast = -K_fast*C_fast
    reaction_slow = -K_slow*C_slow

    lost_fast = lost_soc*p_fast_grid
    lost_slow = lost_soc*(1-p_fast_grid)

    C_fast_new = np.maximum(
        C_fast + (erosion_fast+deposition_fast+vegetation_fast+reaction_fast-lost_fast)*dt,
        0
    )
    C_slow_new = np.maximum(
        C_slow + (erosion_slow+deposition_slow+vegetation_slow+reaction_slow-lost_slow)*dt,
        0
    )
    return C_fast_new, C_slow_new


# =============================================================================
# 9) HELPER: REGRID CLIMATE POINT DATA
# =============================================================================

def create_grid_from_points(lon_points, lat_points, values, gx, gy):
    grid = np.full((len(gy), len(gx)), np.nan)
    for k in range(len(values)):
        j = (np.abs(gx - lon_points[k])).argmin()
        i = (np.abs(gy - lat_points[k])).argmin()
        grid[i,j] = values[k]
    return grid

# =============================================================================
# 10) FIGURE: INITIAL PLOT (SOC)
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots()
cax = ax.imshow(C_fast + C_slow, cmap="viridis",
                extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                origin='upper')
cbar = fig.colorbar(cax, label="SOC (g/kg)")
ax.set_title("Initial SOC Distribution (t=0)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
ax.ticklabel_format(style='plain', axis='x')
plt.savefig(OUTPUT_DIR / "SOC_initial.png")
plt.close()

# =============================================================================
# 11) MAIN SIMULATION LOOP (2007 ~ 2025)
# =============================================================================

CELL_AREA_HA = 100.0
start_year = 2007
end_year   = 2025
global_timestep = 0
M_soil = 1.0e8

C_fast_current = C_fast.copy()
C_slow_current = C_slow.copy()

os.makedirs(OUTPUT_DIR / "Figure", exist_ok=True)
os.makedirs(OUTPUT_DIR / "Data", exist_ok=True)

t_sim_start = time.perf_counter()

for year in range(start_year, end_year+1):
    df_dam_active = df_dam[df_dam["year"] <= year].copy()
    dam_capacity_arr = np.zeros(DEM.shape, dtype=np.float64)
    for _, row in df_dam_active.iterrows():
        i_idx = find_nearest_index(grid_y, row["y"])
        j_idx = find_nearest_index(grid_x, row["x"])
        capacity_10000_m3 = row["capacity_remained"]
        capacity_tons = capacity_10000_m3*10000.0*BULK_DENSITY
        dam_capacity_arr[i_idx,j_idx] = capacity_tons

    nc_file = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{year}.nc"
    if not nc_file.exists():
        print(f"[Warning] NetCDF for year={year} not found => {nc_file}")
        continue

    with nc.Dataset(nc_file) as ds:
        valid_time = ds.variables['valid_time'][:]
        n_time = len(valid_time)
        lon_nc = ds.variables['longitude'][:]
        lat_nc = ds.variables['latitude'][:]
        lai_data = ds.variables['lai_lv'][:]
        tp_data  = ds.variables['tp'][:]

        for month_idx in range(n_time):
            lai_1d = lai_data[month_idx,:]
            LAI_2D = create_grid_from_points(lon_nc, lat_nc, lai_1d, grid_x, grid_y)
            LAI_2D = np.nan_to_num(LAI_2D, nan=np.nanmean(LAI_2D))

            tp_1d = tp_data[month_idx,:]
            tp_1d_mm = tp_1d*1000.0
            RAIN_2D = create_grid_from_points(lon_nc, lat_nc, tp_1d_mm, grid_x, grid_y)
            RAIN_2D = np.nan_to_num(RAIN_2D, nan=np.nanmean(RAIN_2D))

            R_month = calculate_r_factor_monthly(RAIN_2D)
            C_factor_2D = calculate_c_factor(LAI_2D)
            E_t_ha_month = R_month*K_factor*LS_factor*C_factor_2D*P_factor
            E_tcell_month = E_t_ha_month*CELL_AREA_HA

            S = E_tcell_month*(C_fast_current + C_slow_current)
            SOC_loss_g_kg_month = convert_soil_loss_to_soc_loss_monthly(
                E_t_ha_month, (C_fast_current + C_slow_current)
            )

            (D_soil, D_soc,
             inflow_soil, inflow_soc,
             lost_soc) = distribute_soil_and_soc_with_dams_numba(
                E_tcell_month, S, DEM, dam_capacity_arr,
                grid_x, grid_y,
                small_boundary_mask, compute_outlet_mask(small_boundary_mask, DEM),
                large_boundary_mask, compute_outlet_mask(large_boundary_mask, DEM),
                river_mask, sorted_indices
            )

            mean_lost = np.mean(lost_soc)
            max_lost  = np.max(lost_soc)
            min_lost  = np.min(lost_soc)
            print(f"Year={year}, Month={month_idx+1}: Lost_SOC => mean={mean_lost:.3f}, max={max_lost:.3f}, min={min_lost:.3f}")

            V = vegetation_input(LAI_2D)
            C_fast_current, C_slow_current = soc_dynamic_model(
                C_fast_current, C_slow_current,
                SOC_loss_g_kg_month, D_soil, D_soc, V,
                K_fast, K_slow, p_fast_grid,
                dt=1, M_soil=M_soil, lost_soc=lost_soc
            )

            global_timestep+=1
            print(f"[Info] Completed Year={year}, Month={month_idx+1}, Timestep={global_timestep}.\n")

            # 绘图
            fig, ax = plt.subplots()
            cax_ = ax.imshow(C_fast_current + C_slow_current, cmap="viridis",
                             extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                             origin='upper')
            cbar_ = fig.colorbar(cax_, label="SOC (g/kg)")
            ax.set_title(f"SOC at t={global_timestep} (Year={year}, Month={month_idx+1})")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='x')

            out_fig = OUTPUT_DIR / "Figure" / f"SOC_{year}_{month_idx+1:02d}_timestep_{global_timestep}_River.png"
            plt.savefig(out_fig)
            plt.close()

            # Save CSV
            rows_grid, cols_grid = C_fast_current.shape
            lat_list = []
            lon_list = []
            landuse_list = []
            C_fast_list = []
            C_slow_list = []
            erosion_fast_list = []
            erosion_slow_list = []
            deposition_fast_list = []
            deposition_slow_list = []
            vegetation_fast_list = []
            vegetation_slow_list = []
            reaction_fast_list = []
            reaction_slow_list = []
            E_t_ha_list = []
            lost_soc_list_ = []

            for i in range(rows_grid):
                for j in range(cols_grid):
                    cell_lat = grid_y[i]
                    cell_lon = grid_x[j]
                    lat_list.append(cell_lat)
                    lon_list.append(cell_lon)
                    landuse_list.append(str(LANDUSE[i, j]))
                    C_fast_list.append(C_fast_current[i, j])
                    C_slow_list.append(C_slow_current[i, j])

                    erosion_fast_list.append(-SOC_loss_g_kg_month[i, j]*p_fast_grid[i, j])
                    erosion_slow_list.append(-SOC_loss_g_kg_month[i, j]*(1 - p_fast_grid[i, j]))

                    dep_conc = (D_soil[i, j]*1000.0)/M_soil
                    deposition_fast_list.append(dep_conc*p_fast_grid[i, j])
                    deposition_slow_list.append(dep_conc*(1 - p_fast_grid[i, j]))

                    vegetation_fast_list.append(V[i, j]*p_fast_grid[i, j])
                    vegetation_slow_list.append(V[i, j]*(1 - p_fast_grid[i, j]))

                    reaction_fast_list.append(-K_fast[i, j]*C_fast_current[i, j])
                    reaction_slow_list.append(-K_slow[i, j]*C_slow_current[i, j])

                    E_t_ha_list.append(E_t_ha_month[i, j])
                    lost_soc_list_.append(lost_soc[i, j])

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
                'Lost_SOC_River': lost_soc_list_
            })
            out_csv = OUTPUT_DIR / "Data" / f"SOC_terms_{year}_{month_idx+1:02d}_timestep_{global_timestep}_River.csv"
            df_out.to_csv(out_csv, index=False)
            print(f"[Info] Saved CSV => {out_csv}")

sim_time = time.perf_counter() - t_sim_start
print(f"[Done] Simulation complete, total time={sim_time:.2f}s.")
print("Final SOC => C_fast_current + C_slow_current.")
