import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import TeeOutput, gaussian_blur_with_nan, compute_const_RUSLE, print_factor_info
from init import init_global_data_structs, clean_nan, get_PAST_LAI
from paths import Paths 
from config import *
from global_structs import INIT_VALUES, MAP_STATS
from river_basin import precompute_river_basin

def run_model():
    # INIT GRID AND GLOBAL GRID INFO
    print("Initializing global data structures...")
    init_global_data_structs()
    
    # Load future initial data if running future only
    if SKIP_TO_FUTURE and FUTURE_INIT_FILE is not None:
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
    compute_const_RUSLE()
    print_factor_info(INIT_VALUES.LS_FACTOR, name="LS Factor")
    print_factor_info(INIT_VALUES.P_FACTOR, name="P Factor")
    
if __name__ == "__main__":
    with open(Paths.OUTPUT_DIR / "out.log", "w") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Log generated at: {timestamp}\n\n")
        #f.write(get_param_log() + "\n")
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(f, original_stdout)
        try:
            run_model()
        finally:
            sys.stdout = original_stdout