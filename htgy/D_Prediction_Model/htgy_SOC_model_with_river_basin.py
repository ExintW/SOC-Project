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

import geopandas as gpd
from shapely.geometry import Point
from shapely import vectorized  # We use precomputed masks instead of per-point vectorized intersection.
from shapely.prepared import prep
from pathlib import Path
import sys
from numba import njit, prange
import numba
import glob

sys.path.append(os.path.dirname(__file__))

from RUSLE_Calculations import *
from globalss import *
from Init import init_global_data_structs
from River_Basin import * 
from utils import *
from simulation_loop import run_simulation_year

# Append parent directory to path to access 'globals' if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # Expects DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

def run_model(a, b, c, start_year, end_year, past_year, future_year, fraction=1):
    # =============================================================================
    # CSV READING & GRID SETUP & SOC PARTITION
    # =============================================================================
    init_global_data_structs()

    # =============================================================================
    # RASTERIZE RIVER BASIN BOUNDARIES & MAIN RIVER USING PRECOMPUTED MASKS
    # =============================================================================
    precompute_river_basin_1()

    # =============================================================================
    # COMPUTE CONSTANT RUSLE FACTORS
    # =============================================================================
    K_factor = calculate_k_factor(INIT_VALUES.SILT, INIT_VALUES.SAND, INIT_VALUES.CLAY, INIT_VALUES.SOC, INIT_VALUES.LANDUSE)     # constant K factor (not used)
    K_factor = np.nan_to_num(K_factor, nan=np.nanmean(K_factor))
    # K_factor = np.full_like(C, 0.03)
    LS_factor = calculate_ls_factor(INIT_VALUES.SLOPE, INIT_VALUES.DEM)
    LS_factor = resample_LS_to_1km_grid(LS_factor)
    print(f"Total elements in LS: {LS_factor.size}, with max = {np.max(LS_factor)}, min = {np.min(LS_factor)}, and mean = {np.mean(LS_factor)}")
    P_factor = np.array([
        [calculate_p_factor(INIT_VALUES.LANDUSE[i, j], INIT_VALUES.SLOPE[i, j]) for j in range(INIT_VALUES.LANDUSE.shape[1])]
        for i in range(INIT_VALUES.LANDUSE.shape[0])
    ])
    print(f"Total elements in P: {P_factor.size}, with max = {np.max(P_factor)}, min = {np.min(P_factor)}, and mean = {np.mean(P_factor)}")

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
    # FIGURE OUTPUT SETUP & INITIAL PLOT
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
    plt.savefig(os.path.join(OUTPUT_DIR / "Figure"/ "SOC_initial.png"))
    plt.close(fig)

    # =============================================================================
    # MAIN SIMULATION LOOP (MONTHLY)
    # =============================================================================
    step_size = 1   # for quick RUSLE vaidation

    # Initialize current SOC pools=
    MAP_STATS.C_fast_current = INIT_VALUES.C_fast.copy()
    MAP_STATS.C_slow_current = INIT_VALUES.C_slow.copy()

    os.makedirs(OUTPUT_DIR / "Figure", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "Data", exist_ok=True)

    # Delete previous results
    if CLEAN_OUTDIR:
        data_dir = OUTPUT_DIR / "Data"
        for file in glob.glob(str(data_dir / "*.csv")):
            os.remove(file)
        for file in glob.glob(str(data_dir / "*.parquet")):
            os.remove(file)
            
        figure_dir = OUTPUT_DIR / "Figure"
        for file in glob.glob(str(figure_dir / "*.png")):
            os.remove(file)

    t_sim_start = time.perf_counter()
    
    if end_year != None:
        for year in range(start_year, end_year + 1, step_size):
            run_simulation_year(year, LS_factor, P_factor, sorted_indices, a=a, b=b, c=c)

    if future_year != None:
        for year in range(end_year+1, future_year + 1):
            run_simulation_year(year, LS_factor, P_factor, sorted_indices, future=True, a=a, b=b, c=c)

    if end_year != None or future_year != None:
        INIT_VALUES.reset()
        MAP_STATS.reset()
        init_global_data_structs(fraction=fraction)
        precompute_river_basin_1()
        MAP_STATS.C_fast_current = INIT_VALUES.C_fast.copy()
        MAP_STATS.C_slow_current = INIT_VALUES.C_slow.copy()

    if past_year != None:
        if fraction == 1:
            for year in range(start_year-1, past_year-1, -1):
                run_simulation_year(year, LS_factor, P_factor, sorted_indices, past=True, a=a, b=b, c=c)
        else:   # run non-reversed past year simulation with given fraction as init condition
            for year in range(past_year, start_year, step_size):
                run_simulation_year(year, LS_factor, P_factor, sorted_indices, a=a, b=b, c=c)

    print(f"Simulation complete. Total simulation time: {time.perf_counter() - t_sim_start:.2f} seconds.")
    print("Final SOC distribution is in C_fast_current + C_slow_current.")
    
    rmse = np.sqrt(np.nanmean((MAP_STATS.C_fast_current + MAP_STATS.C_slow_current - INIT_VALUES.SOC_valid) ** 2))
    return rmse

if __name__ == "__main__":
    a = -1.8
    b = 1.8
    c = 6
    
    start_year = 2007   # year of init condition
    end_year = 2025     # last year of present  (set to None to disable present year)
    past_year = None    # last year of past     (set to None to disable past year)
    future_year = None  # last year of future   (set to None to disable future year)
    
    fraction = 0.9      # fraction of SOC of past year (set to 1 to disable non-reverse past year simulation)
    
    run_model(a, b, c, start_year, end_year, past_year, future_year, fraction)