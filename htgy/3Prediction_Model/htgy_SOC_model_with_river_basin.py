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
       print("Grid extent:", MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(), MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max())
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

from RUSLE_Calculations import *
from globalss import *
from Init import init
from River_Basin import *
from utils import *

# Append parent directory to path to access 'globals' if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # Expects DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

# =============================================================================
# CSV READING & GRID SETUP & SOC PARTITION
# =============================================================================
init()

# =============================================================================
# RASTERIZE RIVER BASIN BOUNDARIES & MAIN RIVER USING PRECOMPUTED MASKS
# =============================================================================
precompute_river_basin()

# =============================================================================
# COMPUTE CONSTANT RUSLE FACTORS
# =============================================================================
K_factor = calculate_k_factor(INIT_VALUES.SILT, INIT_VALUES.SAND, INIT_VALUES.CLAY, INIT_VALUES.SOC, INIT_VALUES.LANDUSE)     # constant K factor (not used)
K_factor = np.nan_to_num(K_factor, nan=np.nanmean(K_factor))
# K_factor = np.full_like(C, 0.03)
LS_factor = calculate_ls_factor(INIT_VALUES.SLOPE)
print(f"Total elements in LS: {LS_factor.size}, with {np.sum(LS_factor > 50)} elements > 50, and mean = {np.mean(LS_factor)}")
P_factor = np.array([
    [calculate_p_factor(INIT_VALUES.LANDUSE[i, j]) for j in range(INIT_VALUES.LANDUSE.shape[1])]
    for i in range(INIT_VALUES.LANDUSE.shape[0])
])

# =============================================================================
# PRECOMPUTE SORTED INDICES FOR NUMBA ACCELERATION (TO AVOID IN-NUMBA SORTING)
# =============================================================================
rows, cols = INIT_VALUES.DEM.shape
flat_dem = INIT_VALUES.DEM.flatten()
sorted_flat_indices = np.argsort(flat_dem)[::-1]  # descending order
sorted_indices = np.empty((sorted_flat_indices.shape[0], 2), dtype=np.int64)
sorted_indices[:, 0], sorted_indices[:, 1] = np.unravel_index(sorted_flat_indices, (rows, cols))

# Using Numba for acceleration with a fractional flow approach.
try:
    atomic_add = numba.atomic.add
except AttributeError:
    def atomic_add(array, index, value):
        array[index[0], index[1]] += value
    print("Warning: numba.atomic.add not available; using non-atomic addition (serial mode).")

# =============================================================================
# 8) VEGETATION INPUT & UPDATED SOC DYNAMIC MODEL
# =============================================================================
def vegetation_input(LAI):
    """
    Compute vegetation input based on LAI using an empirical formula.
    E.g., V = a * LAI + b
    """
    return 0.05764345 * LAI - 0.00429286

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
# 10) FIGURE OUTPUT SETUP & INITIAL PLOT
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots()
cax = ax.imshow(INIT_VALUES.C_fast + INIT_VALUES.C_slow, cmap="viridis",
                extent=[MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(), MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()],
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
C_fast_current = INIT_VALUES.C_fast.copy()
C_slow_current = INIT_VALUES.C_slow.copy()

os.makedirs(OUTPUT_DIR / "Figure", exist_ok=True)
os.makedirs(OUTPUT_DIR / "Data", exist_ok=True)

t_sim_start = time.perf_counter()

# Precompute sorted indices for Numba function (descending DEM)
rows, cols = INIT_VALUES.DEM.shape
flat_dem = INIT_VALUES.DEM.flatten()
sorted_flat_indices = np.argsort(flat_dem)[::-1]
sorted_indices = np.empty((sorted_flat_indices.shape[0], 2), dtype=np.int64)
sorted_indices[:, 0], sorted_indices[:, 1] = np.unravel_index(sorted_flat_indices, (rows, cols))

for year in range(start_year, end_year + 1):
    # Filter dams built on or before current year
    df_dam_active = MAP_STATS.df_dam[MAP_STATS.df_dam["year"] <= year].copy()
    dam_capacity_arr = np.zeros(INIT_VALUES.DEM.shape, dtype=np.float64)
    for _, row in df_dam_active.iterrows():
        i_idx = find_nearest_index(MAP_STATS.grid_y, row["y"])
        j_idx = find_nearest_index(MAP_STATS.grid_x, row["x"])
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
        tp_data_mm = tp_data * 1000.0
        R_annual = calculate_r_factor_annually(tp_data_mm)
        
        E_month_avg_list = []   # for calculating annual mean for validation
        
        for month_idx in range(n_time):
            # Regrid LAI data
            lai_1d = lai_data[month_idx, :]
            LAI_2D = create_grid_from_points(lon_nc, lat_nc, lai_1d, MAP_STATS.grid_x, MAP_STATS.grid_y)
            LAI_2D = np.nan_to_num(LAI_2D, nan=np.nanmean(LAI_2D))

            # Regrid precipitation and convert to mm
            tp_1d = tp_data[month_idx, :]
            tp_1d_mm = tp_1d * 1000.0
            # RAIN_2D = create_grid_from_points(lon_nc, lat_nc, tp_1d_mm, MAP_STATS.grid_x, MAP_STATS.grid_y)
            # RAIN_2D = np.nan_to_num(RAIN_2D, nan=np.nanmean(RAIN_2D))

            # Compute RUSLE factors
            R_month = get_montly_r_factor(R_annual, tp_1d_mm, tp_data_mm)
            R_month = create_grid_from_points(lon_nc, lat_nc, R_month, MAP_STATS.grid_x, MAP_STATS.grid_y)
            R_month = np.nan_to_num(R_month, nan=np.nanmean(R_month))
            print(f"Total elements in R: {R_month.size}, with {np.sum(R_month > 250)} elements > 250, and mean = {np.mean(R_month)}")
            C_factor_2D = calculate_c_factor(LAI_2D)
            print(f"Total elements in C: {C_factor_2D.size}, with {np.sum(C_factor_2D > 1)} elements > 1, and mean = {np.mean(C_factor_2D)}")
            
            # Calculate monthly K factor
            K_month = calculate_k_factor(INIT_VALUES.SILT, INIT_VALUES.SAND, INIT_VALUES.CLAY, (C_fast_current + C_slow_current), INIT_VALUES.LANDUSE)
            K_month = np.nan_to_num(K_month, nan=np.nanmean(K_month))
            K_month = np.clip(K_month, 0, 0.7)
            print(f"Total elements in K: {K_month.size}, with {np.sum(K_month == 0.7)} elements = 0.7, and mean = {np.mean(K_month)}")

            # Calculate soil loss (t/ha/month) & then per cell
            E_t_ha_month = R_month * K_month * LS_factor * C_factor_2D * P_factor
            print(f"Total elements in E: {E_t_ha_month.size}, with max = {np.max(E_t_ha_month)}, and mean = {np.mean(E_t_ha_month)}")
            E_tcell_month = E_t_ha_month * CELL_AREA_HA
            E_month_avg_list.append(np.mean(E_t_ha_month))

            # Compute SOC mass eroded (kg/cell/month)
            S = E_tcell_month * (C_fast_current + C_slow_current)
            SOC_loss_g_kg_month = convert_soil_loss_to_soc_loss_monthly(
                E_t_ha_month, (C_fast_current + C_slow_current)
            )

            # Call the Numba-accelerated routing function
            D_soil, D_soc, inflow_soil, inflow_soc, lost_soc = distribute_soil_and_soc_with_dams_numba(
                E_tcell_month, S, INIT_VALUES.DEM, dam_capacity_arr, MAP_STATS.grid_x, MAP_STATS.grid_y,
                MAP_STATS.small_boundary_mask, compute_outlet_mask(MAP_STATS.small_boundary_mask, INIT_VALUES.DEM),
                MAP_STATS.large_boundary_mask, compute_outlet_mask(MAP_STATS.large_boundary_mask, INIT_VALUES.DEM),
                MAP_STATS.river_mask, sorted_indices
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
                INIT_VALUES.K_fast, INIT_VALUES.K_slow, MAP_STATS.p_fast_grid,
                dt=1,
                M_soil=M_soil,
                lost_soc=lost_soc
            )

            global_timestep += 1
            print(f"Completed simulation for Year {year}, Month {month_idx+1}")

            # Save figure output
            fig, ax = plt.subplots()
            cax = ax.imshow(C_fast_current + C_slow_current, cmap="viridis",
                            extent=[MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(), MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()],
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
                    cell_lon = MAP_STATS.grid_x[j]
                    cell_lat = MAP_STATS.grid_y[i]
                    lat_list.append(cell_lat)
                    lon_list.append(cell_lon)
                    landuse_list.append(str(INIT_VALUES.LANDUSE[i, j]))
                    C_fast_list.append(C_fast_current[i, j])
                    C_slow_list.append(C_slow_current[i, j])

                    # Erosion
                    erosion_fast_list.append(-SOC_loss_g_kg_month[i, j] * MAP_STATS.p_fast_grid[i, j])
                    erosion_slow_list.append(-SOC_loss_g_kg_month[i, j] * (1 - MAP_STATS.p_fast_grid[i, j]))

                    # Deposition
                    deposition_concentration = (D_soc[i, j] * 1000.0) / M_soil
                    deposition_fast_list.append(deposition_concentration * MAP_STATS.p_fast_grid[i, j])
                    deposition_slow_list.append(deposition_concentration * (1 - MAP_STATS.p_fast_grid[i, j]))

                    # Vegetation
                    vegetation_fast_list.append(V[i, j] * MAP_STATS.p_fast_grid[i, j])
                    vegetation_slow_list.append(V[i, j] * (1 - MAP_STATS.p_fast_grid[i, j]))

                    # Reaction
                    reaction_fast_list.append(-INIT_VALUES.K_fast[i, j] * C_fast_current[i, j])
                    reaction_slow_list.append(-INIT_VALUES.K_slow[i, j] * C_slow_current[i, j])

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
