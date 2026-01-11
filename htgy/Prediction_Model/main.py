import os
import sys
import pandas as pd
import numpy as np
import glob
import time

from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import TeeOutput, gaussian_blur_with_nan, compute_const_RUSLE, print_plot_factor_info, precompute_sorted_indices, save_nc, validate_SOC
from init import init_global_data_structs, clean_nan, get_PAST_LAI
from paths import Paths 
from config import *
from global_structs import INIT_VALUES, MAP_STATS
from river_basin import precompute_river_basin
from simulation_loop import run_simulation_year

from A_Data_visualization.png_to_mp4 import generate_mp4

def run_model():
    # INIT GRID AND GLOBAL GRID INFO
    print("Initializing global data structures...")
    init_global_data_structs()
    
    # Load future initial data if running future only
    if SKIP_TO_FUTURE and FUTURE_INIT_FILE is not None:
        print("Loading future initial data...")
        # Path to the snapshot for December of the present period
        future_initial_file = FUTURE_INIT_FILE
        if future_initial_file.exists():
            df_init = pd.read_parquet(future_initial_file)
            # reshape to original grid shape
            INIT_VALUES.C_fast = df_init['C_fast'].values.reshape(INIT_VALUES.C_fast.shape)
            INIT_VALUES.C_slow = df_init['C_slow'].values.reshape(INIT_VALUES.C_slow.shape)
            MAP_STATS.dam_cur_stored = df_init['dam_cur_stored'].values.reshape(MAP_STATS.dam_cur_stored.shape)
        else:
            print(f"Warning: future initial file not found at {future_initial_file}, using default INIT_VALUES")

    # rasterize river basin boundaries & main river using precomputed masks
    print("Precomputing river basin masks...")
    precompute_river_basin()
    
    # clean up global data: set nan to mean and values outside of border to nan
    clean_nan()
    
    # Save cleaned past SOC for validation
    if VALIDATE_PAST:
        SOC_PAST_Total = INIT_VALUES.SOC_PAST_FAST + INIT_VALUES.SOC_PAST_SLOW
        np.savez_compressed(Paths.OUTPUT_DIR / 'SOC_PAST_Total_cleaned', SOC_PAST_Total)
    
    # use gaussian blur to past if enabled
    if USE_GAUSSIAN_BLUR:
        INIT_VALUES.SOC_PAST_FAST = gaussian_blur_with_nan(INIT_VALUES.SOC_PAST_FAST, sigma=SIGMA)
        INIT_VALUES.SOC_PAST_SLOW = gaussian_blur_with_nan(INIT_VALUES.SOC_PAST_SLOW, sigma=SIGMA)
    
    # Get past LAI for Regularization
    if USE_PAST_LAI_TREND:
        get_PAST_LAI()
        
    # Precompute Constant RUSLE Factors
    print("Precomputing constant RUSLE factors...")
    compute_const_RUSLE()
    print_plot_factor_info(INIT_VALUES.LS_FACTOR, name="LS Factor")
    print_plot_factor_info(INIT_VALUES.P_FACTOR, name="P Factor")
    
    # Precompute Sorted Indices of DEM in Decending Order
    print("Precomputing sorted indices of DEM...")
    precompute_sorted_indices()
    
    # Initialize current SOC pools
    MAP_STATS.C_fast_current = INIT_VALUES.C_fast.copy()
    MAP_STATS.C_slow_current = INIT_VALUES.C_slow.copy()
    MAP_STATS.C_fast_current[~MAP_STATS.border_mask] = np.nan
    MAP_STATS.C_slow_current[~MAP_STATS.border_mask] = np.nan
    
    os.makedirs(Paths.OUTPUT_DIR / "Figure", exist_ok=True)
    os.makedirs(Paths.OUTPUT_DIR / "Data", exist_ok=True)
    
    # Delete previous results
    if CLEAN_OUTDIR:
        print("Cleaning output directory...")
        data_dir = Paths.OUTPUT_DIR / "Data"
        for file in glob.glob(str(data_dir / "*.csv")):
            os.remove(file)
        for file in glob.glob(str(data_dir / "*.parquet")):
            os.remove(file)
            
        figure_dir = Paths.OUTPUT_DIR / "Figure"
        for file in glob.glob(str(figure_dir / "*.png")):
            os.remove(file)
            
    # Plot initial SOC
    print_plot_factor_info(MAP_STATS.C_fast_current + MAP_STATS.C_slow_current, name="Initial_SOC", max_val=30)
    
    # =============================================================================
    # Start Simulation
    # =============================================================================
    
    t_sim_start = time.perf_counter()
    
    if END_YEAR != None:
        # Run Present Simulation
        print("Running present simulation...")
        for year in range(INIT_YEAR, END_YEAR + 1):
            run_simulation_year(year)
        if SAVE_NC:
            save_nc(INIT_YEAR, END_YEAR)
    if FUTURE_YEAR != None:
        # Run Future Simulation
        print("Running future simulation...")
        for year in range(END_YEAR + 1, FUTURE_YEAR + 1):
            run_simulation_year(year)
        if SAVE_NC:
            save_nc(INIT_YEAR, FUTURE_YEAR)
        if PAST_YEAR is None:
            print(f"Generating mp4...")
            generate_mp4(start_year=INIT_YEAR, end_year=FUTURE_YEAR)
    
    # Reset current SOC to initial for past simulation
    MAP_STATS.C_fast_current = INIT_VALUES.C_fast.copy()
    MAP_STATS.C_slow_current = INIT_VALUES.C_slow.copy()
    MAP_STATS.C_fast_current[~MAP_STATS.border_mask] = np.nan
    MAP_STATS.C_slow_current[~MAP_STATS.border_mask] = np.nan
    
    if PAST_YEAR != None:
        # Run Past Simulation
        if RUN_FROM_EQUIL:
            start_year = EQUIL_YEAR - 1
            if FUTURE_YEAR != None or END_YEAR != EQUIL_YEAR:
                # Run present again to get to EQUIL_YEAR state
                print(f"Running present simulation to reach equilibrium year {EQUIL_YEAR}...")
                for year in range(INIT_YEAR, EQUIL_YEAR + 1):
                    run_simulation_year(year)
        else:
            start_year = INIT_YEAR
        print("Running past simulation...")
        for year in range(start_year - 1, PAST_YEAR - 1, -1):
            run_simulation_year(year, past=True)
            
        # Drop elements from 2007 to Equil year to aviod double counting
        if RUN_FROM_EQUIL:
            N_DROP = (EQUIL_YEAR - INIT_YEAR + 1) * 12 
            MAP_STATS.total_C_matrix[-N_DROP:] = []
            MAP_STATS.dam_rem_cap_matrix[-N_DROP:] = []
            MAP_STATS.C_fast_matrix[-N_DROP:] = []
            MAP_STATS.C_slow_matrix[-N_DROP:] = []
            MAP_STATS.active_dam_matrix[-N_DROP:] = []
        
        if SAVE_NC:
            save_nc(PAST_YEAR, INIT_YEAR)
        
        print(f"Generating mp4...")
        if FUTURE_YEAR is None:
            generate_mp4(start_year=PAST_YEAR, end_year=END_YEAR)
        else:
            generate_mp4(start_year=PAST_YEAR, end_year=FUTURE_YEAR)
        
        if VALIDATE_PAST:
            pred_stack = np.stack(MAP_STATS.C_total_Past_Valid_list, axis=0)
            pred_avg = np.nanmean(pred_stack, axis=0)
            np.savez_compressed(Paths.OUTPUT_DIR / 'SOC_Past_Total_predicted', pred_avg)
            validate_SOC(pred_avg, SOC_PAST_Total)

    print(f"Simulation complete. Total simulation time: {time.perf_counter() - t_sim_start:.2f} seconds.")
    
    
if __name__ == "__main__":
    with open(Paths.OUTPUT_DIR / "out.log", "w", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Log generated at: {timestamp}\n\n")
        #f.write(get_param_log() + "\n")
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(f, original_stdout)
        try:
            run_model()
        finally:
            sys.stdout = original_stdout