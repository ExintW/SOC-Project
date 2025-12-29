import os
import sys
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import TeeOutput
from init import init_global_data_structs 
from paths import Paths 
from config import *
from global_structs import INIT_VALUES, MAP_STATS
from river_basin import precompute_river_basin

def run_model():
    # =============================================================================
    # INIT GRID AND GLOBAL GRID INFO
    # =============================================================================
    print("Initializing global data structures...")
    init_global_data_structs()
    
    # =============================================================================
    # Load future initial data if running future only
    # =============================================================================
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

    # =============================================================================
    # RASTERIZE RIVER BASIN BOUNDARIES & MAIN RIVER USING PRECOMPUTED MASKS
    # =============================================================================
    precompute_river_basin()
    

if __name__ == "__main__":
    with open(Paths.OUTPUT_DIR / "out.log", "w") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Log generated at: {timestamp}\n\n")
        f.write(get_param_log() + "\n")
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(f, original_stdout)
        try:
            run_model()
        finally:
            sys.stdout = original_stdout