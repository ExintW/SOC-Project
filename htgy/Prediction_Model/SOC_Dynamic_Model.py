import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import Paths 
from global_structs import MAP_STATS, INIT_VALUES
from config import *

def soc_dynamic_model(E_tcell, A, V, month, year, past=False, LAI_avg=None):
    dt = -1 if past else 1
    V *= V_FACTOR
    V = np.clip(V, V_MIN_CLIP, None)
    
    # Load global variables for faster access
    C_fast_current = MAP_STATS.C_fast_current
    C_slow_current = MAP_STATS.C_slow_current
    river_mask = MAP_STATS.river_mask
    low_point_mask = MAP_STATS.low_mask
    low_point_capacity = MAP_STATS.Low_Point_Capacity
    low_point_cur_stored = np.zeros(INIT_VALUES.DEM.shape, dtype=np.float64)
    dam_rem_cap = np.zeros(INIT_VALUES.DEM.shape, dtype=int)
    K_fast = INIT_VALUES.K_fast
    K_slow = INIT_VALUES.K_slow
    DEM = INIT_VALUES.DEM
    small_boundary_mask = MAP_STATS.small_boundary_mask
    small_outlet_mask = MAP_STATS.small_outlet_mask
    large_boundary_mask = MAP_STATS.large_boundary_mask
    large_outlet_mask = MAP_STATS.large_outlet_mask
    loess_border_mask = MAP_STATS.border_mask

    soc_prev_fast = MAP_STATS.C_fast_prev
    soc_prev_slow = MAP_STATS.C_slow_prev
    
    shape = DEM.shape
    dep_soil = np.zeros(shape, np.float64)
    dep_soc_fast   = np.zeros(shape, np.float64)
    dep_soc_slow   = np.zeros(shape, np.float64)
    ero_soil  = np.zeros(shape, np.float64)
    ero_soc   = np.zeros(shape, np.float64)
    lost_soc = np.zeros(shape, dtype=np.float64)      # keep track of river losses
    
    # For Regularization
    max_A = np.nanmax(A)
    max_k_fast = np.nanmax(K_fast)
    max_k_slow = np.nanmax(K_slow)
    MAX_V = np.nanmax(V)
    
    if past:
        if USE_PAST_EQUIL:
            soc_past_fast = INIT_VALUES.SOC_PAST_FAST
            soc_past_slow = INIT_VALUES.SOC_PAST_SLOW
        if RUN_FROM_EQUIL:
            soc_equil_fast = MAP_STATS.C_fast_equil_list[month]
            soc_equil_slow = MAP_STATS.C_slow_equil_list[month]
            
        L_fast = np.zeros(shape, dtype=np.float64)
        L_slow = np.zeros(shape, dtype=np.float64)

        C_fast_past = np.zeros(shape, np.float64)
        C_slow_past = np.zeros(shape, np.float64)
        C_fast_past[~MAP_STATS.loess_border_mask] = np.nan
        C_slow_past[~MAP_STATS.loess_border_mask] = np.nan
        
    else:
        del_soc_fast = np.zeros(shape, dtype=np.float64) # Change in SOC
        del_soc_slow = np.zeros(shape, dtype=np.float64) # Change in SOC
        del_soc_fast[~MAP_STATS.border_mask] = np.nan
        del_soc_slow[~MAP_STATS.border_mask] = np.nan