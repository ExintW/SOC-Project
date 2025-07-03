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
from datetime import datetime

import geopandas as gpd
from shapely.geometry import Point
from shapely import vectorized  # We use precomputed masks instead of per-point vectorized intersection.
from shapely.prepared import prep
from pathlib import Path
import sys
from numba import njit, prange
import numba
import glob
import pandas as pd
import xarray as xr

sys.path.append(os.path.dirname(__file__))

from RUSLE_Calculations import *
from globalss import *
from Init import init_global_data_structs, clean_nan, precompute_low_point
from River_Basin import * 
from utils import *
from simulation_loop import run_simulation_year
from shapely.geometry import LineString, MultiLineString

# Append parent directory to path to access 'globals' if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # Expects DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

def run_model(a, b, c, start_year, end_year, past_year, future_year, fraction=1):
    # =============================================================================
    # CSV READING & GRID SETUP & SOC PARTITION
    # =============================================================================
    init_global_data_structs()

    # =============================================================================
    # LOAD FUTURE-INITIAL SOC FROM PARQUET
    # =============================================================================
    if future_year != None:
        # Path to the snapshot for December of the present period
        future_initial_file = OUTPUT_DIR / "Data" / "SOC_Present 5" / "SOC_terms_2024_12_River.parquet"
        if future_initial_file.exists():
            df_init = pd.read_parquet(future_initial_file)
            # reshape to original grid shape
            INIT_VALUES.C_fast = df_init['C_fast'].values.reshape(INIT_VALUES.C_fast.shape)
            INIT_VALUES.C_slow = df_init['C_slow'].values.reshape(INIT_VALUES.C_slow.shape)
        else:
            print(f"Warning: future initial file not found at {future_initial_file}, using default INIT_VALUES")

    # =============================================================================
    # RASTERIZE RIVER BASIN BOUNDARIES & MAIN RIVER USING PRECOMPUTED MASKS
    # =============================================================================
    precompute_river_basin_1()

    # =============================================================================
    # CLEAN UP GLOBAL DATA: SET NAN TO MEAN AND VALUES OUTSIDE OF BORDER TO NAN
    # =============================================================================
    clean_nan()

    # =============================================================================
    # COMPUTE CONSTANT LOW POINT MASK AND LOW POINT CAPACITY
    # =============================================================================
    MAP_STATS.low_mask, MAP_STATS.Low_Point_Capacity, MAP_STATS.Low_Point_DEM_Dif = precompute_low_point()
    print(f"Low point DEM difference: max = {np.nanmax(MAP_STATS.Low_Point_DEM_Dif):.2f}, min = {np.nanmin(MAP_STATS.Low_Point_DEM_Dif):.2f}, and mean = {np.nanmean(MAP_STATS.Low_Point_DEM_Dif):.2f}")
    
    # =============================================================================
    # COMPUTE CONSTANT RUSLE FACTORS
    # =============================================================================
    K_factor = calculate_k_factor(INIT_VALUES.SILT, INIT_VALUES.SAND, INIT_VALUES.CLAY, INIT_VALUES.SOC, INIT_VALUES.LANDUSE)     # constant K factor (not used)
    K_factor = np.nan_to_num(K_factor, nan=np.nanmean(K_factor))
    # K_factor = np.full_like(C, 0.03)
    K_factor[~MAP_STATS.loess_border_mask] = np.nan
    LS_factor = calculate_ls_factor(INIT_VALUES.SLOPE, INIT_VALUES.DEM)
    LS_factor = resample_LS_to_1km_grid(LS_factor)
    LS_factor[~MAP_STATS.loess_border_mask] = np.nan
    print(f"Total elements in LS: {LS_factor.size}, with max = {np.nanmax(LS_factor)}, min = {np.nanmin(LS_factor)}, and mean = {np.nanmean(LS_factor)}")
    P_factor = np.array([
        [calculate_p_factor(INIT_VALUES.LANDUSE[i, j], INIT_VALUES.SLOPE[i, j]) for j in range(INIT_VALUES.LANDUSE.shape[1])]
        for i in range(INIT_VALUES.LANDUSE.shape[0])
    ])
    P_factor[~MAP_STATS.loess_border_mask] = np.nan
    print(f"Total elements in P: {P_factor.size}, with max = {np.nanmax(P_factor)}, min = {np.nanmin(P_factor)}, and mean = {np.nanmean(P_factor)}")

    # =============================================================================
    # PRECOMPUTE SORTED INDICES FOR SOC SIMULATION
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
    # MAIN SIMULATION LOOP (MONTHLY)
    # =============================================================================
    step_size = 1   # for quick RUSLE vaidation

    # Initialize current SOC pools=
    MAP_STATS.C_fast_current = INIT_VALUES.C_fast.copy()
    MAP_STATS.C_slow_current = INIT_VALUES.C_slow.copy()
    MAP_STATS.C_fast_current[~MAP_STATS.loess_border_mask] = np.nan
    MAP_STATS.C_slow_current[~MAP_STATS.loess_border_mask] = np.nan

    os.makedirs(OUTPUT_DIR / "Figure", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "Data", exist_ok=True)

    # Initialize current dam capacity
    MAP_STATS.dam_cur_stored = np.zeros(INIT_VALUES.DEM.shape, dtype=np.float64)

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

    # =============================================================================
    # FIGURE OUTPUT SETUP & INITIAL PLOT
    # =============================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots()
    cax = ax.imshow(INIT_VALUES.C_fast + INIT_VALUES.C_slow, cmap="viridis", vmin=0,vmax=30,
                    extent=[MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(), MAP_STATS.grid_y.min(),
                            MAP_STATS.grid_y.max()],
                    origin='upper')
    # overlay the border (no fill, just outline)
    border = MAP_STATS.loess_border_geom.boundary

    if isinstance(border, LineString):
        x, y = border.xy
        ax.plot(x, y, color="black", linewidth=0.4)
    elif isinstance(border, MultiLineString):
        for seg in border.geoms:
            x, y = seg.xy
            ax.plot(x, y, color="black", linewidth=0.4)
    cbar = fig.colorbar(cax, label="SOC (g/kg)")
    ax.set_title("Initial SOC Distribution (t = 0)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='x')
    plt.savefig(os.path.join(OUTPUT_DIR / "Figure" / "SOC_initial.png"), dpi=600)
    plt.close(fig)

    t_sim_start = time.perf_counter()
    
    if end_year != None:
        for year in range(start_year, end_year + 1, step_size):
            run_simulation_year(year, LS_factor, P_factor, sorted_indices, a=a, b=b, c=c)
        if SAVE_NC:
            # stack into an (X, 844, 1263) array
            Fast_C_array = np.stack(MAP_STATS.C_fast_matrix, axis=0)
            Slow_C_array = np.stack(MAP_STATS.C_slow_matrix, axis=0)
            total_C_array = np.stack(MAP_STATS.total_C_matrix, axis=0)
            active_dam_arr = np.stack(MAP_STATS.active_dam_matrix, axis=0)
            np.savez(
                os.path.join(OUTPUT_DIR, f"Fast SOC year {start_year}-{end_year}.npz"),
                soc_fast=Fast_C_array
            )
            np.savez(
                os.path.join(OUTPUT_DIR, f"Slow SOC year {start_year}-{end_year}.npz"),
                soc_slow=Slow_C_array
            )
            print(f"Saved slow-C matrix from year {start_year}-{end_year} of shape {Slow_C_array.shape}")
            print(f"Saved fast-C matrix from year {start_year}-{end_year} of shape {Fast_C_array.shape}")
            # ─── SAVE active_dam_matrix as .npz ───────────────────────────────────
            np.savez(
                os.path.join(OUTPUT_DIR, f"Active_dams_{start_year}-{end_year}.npz"),
                check_dams=active_dam_arr
            )
            print(f"Saved active-dam matrix from year {start_year}-{end_year} of shape {active_dam_arr.shape}")
            # ─── SAVE static DEM (no time axis) ────────────────────────────────────
            dem_array = INIT_VALUES.DEM.copy()
            np.savez(
                os.path.join(OUTPUT_DIR, "DEM.npz"),
                dem=dem_array
            )
            print(f"Saved DEM of shape {dem_array.shape} to DEM.npz")
            # ───────────────────────────────────────────────────────────────────────
            nc_path = OUTPUT_DIR / f"Total_C_{start_year}-{end_year}_monthly.nc"
            export_total_C_netcdf(
                total_C_array,
                time_start=start_year,
                lat=MAP_STATS.grid_y,
                lon=MAP_STATS.grid_x,
                out_path=nc_path
            )

    if future_year != None:
        for year in range(start_year, future_year + 1):
            run_simulation_year(year, LS_factor, P_factor, sorted_indices, future=True, a=a, b=b, c=c)
        # stack into an (X, 844, 1263) array
        total_C_array = np.stack(MAP_STATS.total_C_matrix, axis=0)
        Fast_C_array = np.stack(MAP_STATS.C_fast_matrix, axis=0)
        Slow_C_array = np.stack(MAP_STATS.C_slow_matrix, axis=0)
        active_dam_arr = np.stack(MAP_STATS.active_dam_matrix, axis=0)
        np.savez(
            os.path.join(OUTPUT_DIR, f"Fast SOC year {start_year}-{future_year}.npz"),
            soc_fast=Fast_C_array
        )
        np.savez(
            os.path.join(OUTPUT_DIR, f"Slow SOC year {start_year}-{future_year}.npz"),
            soc_slow=Slow_C_array
        )
        print(f"Saved slow-C matrix from year {start_year}-{future_year} of shape {Slow_C_array.shape}")
        print(f"Saved fast-C matrix from year {start_year}-{future_year} of shape {Fast_C_array.shape}")
        np.savez(
            os.path.join(OUTPUT_DIR, f"Active_dams_{start_year}-{future_year}.npz"),
            check_dams=active_dam_arr
        )
        print(f"Saved active-dam matrix from year {start_year}-{future_year} of shape {active_dam_arr.shape}")
        # ─────────────────────────────────────────────────────────────────────
        # ─── SAVE static DEM (no time axis) ────────────────────────────────────
        dem_array = INIT_VALUES.DEM.copy()
        np.savez(
            os.path.join(OUTPUT_DIR, "DEM.npz"),
            dem=dem_array
        )
        print(f"Saved DEM of shape {dem_array.shape} to DEM.npz")
        # ───────────────────────────────────────────────────────────────────────
        nc_path = OUTPUT_DIR / f"Total_C_{start_year}-{future_year}_monthly.nc"
        export_total_C_netcdf(
            total_C_array,
            time_start=start_year,
            lat=MAP_STATS.grid_y,
            lon=MAP_STATS.grid_x,
            out_path=nc_path
        )


    # if end_year != None or future_year != None:
    #     # INIT_VALUES.reset()
    #     # MAP_STATS.reset()
    #     init_global_data_structs(fraction=fraction)
    #     clean_nan()
    #     # precompute_river_basin_1()
    #     MAP_STATS.C_fast_current = INIT_VALUES.C_fast.copy()
    #     MAP_STATS.C_slow_current = INIT_VALUES.C_slow.copy()
    #     MAP_STATS.C_fast_current[~MAP_STATS.loess_border_mask] = np.nan
    #     MAP_STATS.C_slow_current[~MAP_STATS.loess_border_mask] = np.nan

    if end_year == None or not RUN_FROM_EQUIL:    # for running from equilibrium
        if future_year == None:
            end_year = start_year - 1
        
    if past_year != None:
        if fraction == 1:
            # for year in range(start_year-1, past_year-1, -1):
            for year in range(end_year, past_year-1, -1):
                run_simulation_year(year, LS_factor, P_factor, sorted_indices, past=True, a=a, b=b, c=c)
            # stack into an (X, 844, 1263) array
            total_C_array = np.stack(MAP_STATS.total_C_matrix, axis=0)
            Fast_C_array = np.stack(MAP_STATS.C_fast_matrix, axis=0)
            Slow_C_array = np.stack(MAP_STATS.C_slow_matrix, axis=0)
            active_dam_arr = np.stack(MAP_STATS.active_dam_matrix, axis=0)

            np.savez(
                os.path.join(OUTPUT_DIR, f"Fast SOC year {past_year}-{end_year}.npz"),
                soc_fast=Fast_C_array
            )
            np.savez(
                os.path.join(OUTPUT_DIR, f"Slow SOC year {past_year}-{end_year}.npz"),
                soc_slow=Slow_C_array
            )
            print(f"Saved slow-C matrix from year {past_year}-{end_year} of shape {Slow_C_array.shape}")
            print(f"Saved fast-C matrix from year {past_year}-{end_year} of shape {Fast_C_array.shape}")

            np.savez(
                os.path.join(OUTPUT_DIR, f"Active_dams_{past_year}-{end_year}.npz"),
                check_dams=active_dam_arr
            )
            print(f"Saved active-dam matrix from year {past_year}-{end_year} of shape {active_dam_arr.shape}")
            # ─── SAVE static DEM (no time axis) ────────────────────────────────────
            dem_array = INIT_VALUES.DEM.copy()
            np.savez(
                os.path.join(OUTPUT_DIR, "DEM.npz"),
                dem=dem_array
            )
            print(f"Saved DEM of shape {dem_array.shape} to DEM.npz")
            # ───────────────────────────────────────────────────────────────────────

            nc_path = OUTPUT_DIR / f"Total_C_{past_year}-{start_year}_monthly.nc"
            export_total_C_netcdf(
                total_C_array,
                time_start=past_year,
                lat=MAP_STATS.grid_y,
                lon=MAP_STATS.grid_x,
                out_path=nc_path
            )
        else:   # run non-reversed past year simulation with given fraction as init condition
            for year in range(past_year, start_year, step_size):
                run_simulation_year(year, LS_factor, P_factor, sorted_indices, a=a, b=b, c=c)

    print(f"Simulation complete. Total simulation time: {time.perf_counter() - t_sim_start:.2f} seconds.")
    print("Final SOC distribution is in C_fast_current + C_slow_current.")
    
    rmse = np.sqrt(np.nanmean((MAP_STATS.C_fast_current + MAP_STATS.C_slow_current - INIT_VALUES.SOC_valid) ** 2))
    print(f"RMSE = {rmse}")
    return rmse

if __name__ == "__main__":
    a = -1.9
    b = 1.78
    c = 5.5
    
    start_year =  2025  # year of init condition, default is 2007, set to 2025 for future
    end_year = None    # last year of present  (set to None to disable present year)
    past_year = None    # last year of past     (set to None to disable past year)
    future_year = 2100  # last year of future   (set to None to disable future year)
    
    fraction = 1      # fraction of SOC of past year (set to 1 to disable non-reverse past year simulation)
    
    log = True     # save output to a log file
    
    if log:
        with open(OUTPUT_DIR / "out.log", "w") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Log generated at: {timestamp}\n\n")
            f.write(get_param_log() + "\n")
            original_stdout = sys.stdout
            sys.stdout = f
            try:
                rmse = run_model(a, b, c, start_year, end_year, past_year, future_year, fraction)
            finally:
                sys.stdout = original_stdout
    else:
        rmse = run_model(a, b, c, start_year, end_year, past_year, future_year, fraction)