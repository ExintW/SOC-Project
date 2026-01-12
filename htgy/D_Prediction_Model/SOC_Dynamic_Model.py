import os
import sys
import numpy as np
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import Paths 
from global_structs import MAP_STATS, INIT_VALUES
from config import *
from utils import plot_SOC

_NEIGHBOR_OFFSETS = ((-1, -1), (-1, 0), (-1, 1),
                    ( 0, -1),          ( 0, 1),
                    ( 1, -1), ( 1, 0), ( 1, 1))
_NEIGHBOR_DIST   = (math.sqrt(2), 1.0, math.sqrt(2),
                    1.0,                        1.0,
                    math.sqrt(2), 1.0, math.sqrt(2))

def soc_dynamic_model(E_tcell, A, V, month, year, past=False, LAI_avg=None):
    """
    SOC Dynamic Model
    
    Forawrd Simulation:
        dSOC/dt = -k*SOC + V - A*SOC + D
    Reverse Simulation:
        SOC_t-1 = (SOC_t - V_t-1 - D_t-1) / (1 - k - A_t-1)
    Reverse with Regularization:
        SOC_t-1 = ((1 - k - A_t-1) * (SOC_t - V_t-1 - D_t-1) + lambda^2 * SOC_prior) / ((1 - k - A_t-1)^2 + lambda^2)
    Where D = SUM_neighbors(A_neighbor * SOC_neighbor)
    """
    dt = -1 if past else 1
    V *= V_FACTOR
    V = np.clip(V, V_MIN_CLIP, None)
    A = np.clip(A, 0, A_MAX)
    
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
    border_mask = MAP_STATS.border_mask
    
    active_dams = MAP_STATS.active_dams
    dam_cur_stored = MAP_STATS.dam_cur_stored
    dam_max_cap = MAP_STATS.dam_max_cap
    full_dams = MAP_STATS.full_dams

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
        C_fast_past[~MAP_STATS.border_mask] = np.nan
        C_slow_past[~MAP_STATS.border_mask] = np.nan
        C_equil_fast = np.zeros(shape, np.float64)
        C_equil_slow = np.zeros(shape, np.float64)
        
    else:
        del_soc_fast = np.zeros(shape, dtype=np.float64) # Change in SOC
        del_soc_slow = np.zeros(shape, dtype=np.float64) # Change in SOC
        del_soc_fast[~MAP_STATS.border_mask] = np.nan
        del_soc_slow[~MAP_STATS.border_mask] = np.nan
        
    # Start Loop through all grid cells from highest to lowest DEM
    prev_dem = np.inf
    for point in INIT_VALUES.SORTED_INDICES:
        row = point[0]
        col = point[1]
        
        if not border_mask[row, col]:
            continue
        
        assert DEM[row][col] <= prev_dem, f"Error: DEM not in decending order! cur_dem = {DEM[row][col]}, prev_dem = {prev_dem}"
        prev_dem= DEM[row][col]
        
        cur_fast_proportion = C_fast_current[row][col] / (C_slow_current[row][col] + C_fast_current[row][col] + 1e-9)
        cur_slow_proportion = C_slow_current[row][col] / (C_slow_current[row][col] + C_fast_current[row][col] + 1e-9)
        
        # Check if point is river
        if river_mask[row][col]:
            lost_soc[row][col] += dep_soc_fast[row][col] + dep_soc_slow[row][col]
            C_fast_current[row][col] = 0.0
            C_slow_current[row][col] = 0.0
            continue
        
        # Check if point is low point
        if low_point_mask[row][col] and not active_dams[row][col]:
            ero_soc[row][col] = 0.0
            ero_soil[row][col] = 0.0
            if low_point_cur_stored[row][col] <= low_point_capacity[row][col]:
                low_point_cur_stored[row][col] += dep_soil[row][col]
            else:
                dep_soc_fast[row][col] = 0.0
                dep_soc_slow[row][col] = 0.0
                dep_soil[row][col] = 0.0
        
        if past:
            L_fast[row][col] = 1 - K_fast[row][col]
            L_slow[row][col] = 1 - K_slow[row][col]
            ero_soc[row][col] = A[row][col].copy()
        else:
            ero_soc[row][col] = A[row][col] * (C_fast_current[row][col] + C_slow_current[row][col])
        
        dam_proportion = 1
        
        if active_dams[row][col]:    # is dam
            if dam_cur_stored[row][col] <= dam_max_cap[row][col]:
                ero_soc[row][col] = 0.0
                ero_soil[row][col] = 0.0
                dam_proportion = 0.0
            else:
                full_dams[row][col] = 1
                extra_soil = dam_cur_stored[row][col] - dam_max_cap[row][col]
                assert extra_soil > 0, f'Error: dam extra soil <= 0: dam_cur_stored[row][col] = {dam_cur_stored[row][col]}, dam_max_cap[row][col] = {dam_max_cap[row][col]}'
                ero_proportion = extra_soil / dam_cur_stored[row][col]
                assert ero_proportion <= 1, f'Error: extra soil > total soil in dam: extra_soil = {extra_soil}, dam_cur_stored[row][col] = {dam_cur_stored[row][col]}, dam_max_cap[row][col] = {dam_max_cap[row][col]}'
                ero_soc[row][col] = A[row][col] * ero_proportion
                if not past:
                    ero_soc[row][col] *= (C_fast_current[row][col] + C_slow_current[row][col])   # Only extra soc (over capacity) will erode
                ero_soil[row][col] = ero_proportion * E_tcell[row][col]    # Only extra soil (over capacity) will erode
                dam_proportion = ero_proportion
            dam_cur_stored[row][col] += dt * dep_soil[row][col]
            dam_cur_stored[row][col] -= dt * ero_soil[row][col]
            dam_cur_stored[row][col] = max(0,dam_cur_stored[row][col])
            dam_rem_cap[row][col] = max(0,dam_max_cap[row][col] - dam_cur_stored[row][col])
        
        if past:
            L_fast[row][col] -= ero_soc[row][col]
            L_slow[row][col] -= ero_soc[row][col]
            
            L_fast[row][col] = max(L_fast[row][col], L_FAST_MIN)
            L_slow[row][col] = max(L_slow[row][col], L_SLOW_MIN)
            
            C_fast_past[row][col] = C_fast_current[row][col] - (V_FAST_PROP * V[row][col])
            C_fast_past[row][col] /= L_fast[row][col] + 1e-9
            # C_slow_past[row][col] = C_slow_current[row][col] - (init_slow_proportion[row][col] * V[row][col])
            C_slow_past[row][col] = C_slow_current[row][col] - (1 - V_FAST_PROP) * V[row][col]
            C_slow_past[row][col] /= L_slow[row][col] + 1e-9
            
            if dep_soc_fast[row][col] > 0.0:
                C_fast_past[row][col] -= dep_soc_fast[row][col] / (L_fast[row][col] + 1e-9)
            if dep_soc_slow[row][col] > 0.0:
                C_slow_past[row][col] -= dep_soc_slow[row][col] / (L_slow[row][col] + 1e-9)
                
            C_fast_past[row][col] = max(C_fast_past[row][col], 0)
            # for humification
            C_slow_past[row][col] -= (ALPHA * C_fast_past[row][col] * K_fast[row][col]) / (L_slow[row][col] + 1e-9)
            C_slow_past[row][col] = max(C_slow_past[row][col], 0)

            if USE_TIKHONOV and MAP_STATS.REG_counter == 1:
                if USE_SPATIAL_REG:
                    if USE_K_FOR_SPATIAL:
                        reg_const_fast = REG_CONST_BASE * (1 + REG_ALPHA * (K_fast[row][col] / (max_k_fast + 1e-9)))
                        reg_const_slow = REG_CONST_BASE * (1 + REG_ALPHA * (K_slow[row][col] / (max_k_slow + 1e-9)))
                    else:
                        reg_const_fast = REG_CONST_BASE * (1 + REG_ALPHA * (A[row][col] / (max_A + 1e-9)))
                        reg_const_slow = REG_CONST_BASE * (1 + REG_ALPHA * (A[row][col] / (max_A + 1e-9)))
                    if ADD_V_IN_SPATIAL:
                        reg_const_fast += REG_CONST_BASE * REG_BETA * (1 - (V[row][col] / MAX_V))
                        reg_const_slow += REG_CONST_BASE * REG_BETA * (1 - (V[row][col] / MAX_V))
                else:
                    reg_const_fast = REG_CONST
                    reg_const_slow = REG_CONST
                if USE_PAST_EQUIL:
                    if USE_PAST_EQUIL_AVG:
                        C_equil_fast[row][col] = (soc_past_fast[row][col] + soc_equil_fast[row][col]) / 2
                        C_equil_slow[row][col] = (soc_past_slow[row][col] + soc_equil_slow[row][col]) / 2
                    elif USE_PAST_EQUIL_PREV_AVG:
                        C_equil_fast[row][col] = (soc_past_fast[row][col] + soc_equil_fast[row][col] + soc_prev_fast[row][col]) / 3
                        C_equil_slow[row][col] = (soc_past_slow[row][col] + soc_equil_slow[row][col] + soc_prev_slow[row][col]) / 3
                    elif USE_DYNAMIC_AVG and not ALWAYS_USE_PAST:
                        if year < PAST_KNOWN:
                            # if less than PAST_KNOWN: use PAST_KNOWN with LAI trend as prior
                            if USE_PAST_LAI_TREND:
                                C_equil_fast[row][col] = soc_past_fast[row][col] * (LAI_avg / INIT_VALUES.LAI_PAST[month])
                                C_equil_slow[row][col] = soc_past_slow[row][col] * (LAI_avg / INIT_VALUES.LAI_PAST[month])
                            else:
                                C_equil_fast[row][col] = soc_past_fast[row][col]
                                C_equil_slow[row][col] = soc_past_slow[row][col]
                        else:
                            w_equil = (year - PAST_KNOWN) / (EQUIL_YEAR - PAST_KNOWN)
                            w_PAST_KNOWN = 1 - w_equil
                            C_equil_fast[row][col] = w_PAST_KNOWN * soc_past_fast[row][col] + w_equil * soc_equil_fast[row][col]
                            C_equil_slow[row][col] = w_PAST_KNOWN * soc_past_slow[row][col] + w_equil * soc_equil_slow[row][col]
                        
                    elif abs(year - PAST_KNOWN) < abs(year - EQUIL_YEAR) or ALWAYS_USE_PAST:
                        C_equil_fast[row][col] = soc_past_fast[row][col]
                        C_equil_slow[row][col] = soc_past_slow[row][col]
                    else:
                        C_equil_fast[row][col] = soc_equil_fast[row][col]
                        C_equil_slow[row][col] = soc_equil_slow[row][col]
                    
                    if USE_PAST_LAI_TREND and (abs(year - PAST_KNOWN) < abs(year - EQUIL_YEAR) or ALWAYS_USE_PAST) and not USE_DYNAMIC_AVG:
                        C_equil_fast[row][col] = soc_past_fast[row][col] * (LAI_avg / INIT_VALUES.LAI_PAST[month])
                        C_equil_slow[row][col] = soc_past_slow[row][col] * (LAI_avg / INIT_VALUES.LAI_PAST[month])
                else:
                    C_equil_fast[row][col] = soc_equil_fast[row][col]
                    C_equil_slow[row][col] = soc_equil_slow[row][col]
                if USE_PRIOR_PREV_AVG:
                    C_equil_fast[row][col] = (C_equil_fast[row][col] + soc_prev_fast[row][col]) / 2
                    C_equil_slow[row][col] = (C_equil_slow[row][col] + soc_prev_slow[row][col]) / 2
                        
                C_fast_past[row][col] = ((L_fast[row][col] ** 2) * C_fast_past[row][col]) + (reg_const_fast * C_equil_fast[row][col])
                C_fast_past[row][col] /= (L_fast[row][col] ** 2) + reg_const_fast
                C_slow_past[row][col] = ((L_slow[row][col] ** 2) * C_slow_past[row][col]) + (reg_const_slow * C_equil_slow[row][col])
                C_slow_past[row][col] /= (L_slow[row][col] ** 2) + reg_const_slow
        
        if not past:
            del_soc_fast[row][col] += dep_soc_fast[row][col] - cur_fast_proportion * ero_soc[row][col] - (K_fast[row][col] * C_fast_current[row][col]) + V_FAST_PROP * V[row][col]
            del_soc_slow[row][col] += dep_soc_slow[row][col] - cur_slow_proportion * ero_soc[row][col] - (K_slow[row][col] * C_slow_current[row][col]) + (1 - V_FAST_PROP) * V[row][col] + ALPHA * K_fast[row][col] * C_fast_current[row][col]
        
        if dam_proportion > 0:
            if past:
                C_fast = C_fast_past[row][col]
                C_slow = C_slow_past[row][col]
            else:
                C_fast = C_fast_current[row][col]
                C_slow = C_slow_current[row][col]
            get_deposition_of_point(E_tcell, A, point, dep_soil, dep_soc_fast, dep_soc_slow, DEM,
                                    C_fast * dam_proportion, C_slow * dam_proportion,
                                    low_point_cur_stored, low_point_capacity,
                                    small_boundary_mask, small_outlet_mask,
                                    large_boundary_mask, large_outlet_mask,
                                    border_mask)
    
    if past and USE_TIKHONOV and MAP_STATS.REG_counter == 1:
        MAP_STATS.REG_counter = REG_FREQ
        if LAI_avg is not None:
            print(f"LAI Proportion: {LAI_avg / INIT_VALUES.LAI_PAST[month]}")
            
        if PLOT_PRIOR:
            C_equil_total = C_equil_fast + C_equil_slow
            C_equil_total[~MAP_STATS.border_mask] = np.nan
            plot_SOC(C_equil_total, year, month, ext='Prior')
    elif past and USE_TIKHONOV:
        MAP_STATS.REG_counter -= 1
        
    MAP_STATS.dam_cur_stored = dam_cur_stored
    MAP_STATS.full_dams = full_dams
    MAP_STATS.dam_rem_cap = dam_rem_cap
    
    # =============================================================================
    # For Debugging
    # =============================================================================
    if PRINT_MAX:
        if past:
            max_idx = np.unravel_index(np.nanargmax(C_fast_past), C_fast_past.shape)
            row = max_idx[0]
            col = max_idx[1]
            print(f'idx = {max_idx}')
            print(f'K_fast = {K_fast[row][col]}')
            print(f'C_fast_current = {C_fast_current[row][col]}')
            print(f'C_fast_past = {C_fast_past[row][col]}')
            print(f'K_slow = {K_slow[row][col]}')
            print(f'dep_soc_fast = {dep_soc_fast[row][col]}')
            print(f'dep_soc_slow = {dep_soc_slow[row][col]}')
            print(f'ero_soc = {ero_soc[row][col] * (C_fast_past[row][col])}')
            print(f'A = {A[row][col]}')
            print(f'V = {V[row][col] * V_FAST_PROP}')
            print(f'L_fast = {L_fast[row][col]}')
            print(f'humification = {ALPHA * K_fast[row][col] * C_fast_past[row][col]}')
            print('-----------------------------------------------------------------------')
            max_idx = np.unravel_index(np.nanargmax(C_slow_past), C_slow_past.shape)
            row = max_idx[0]
            col = max_idx[1]
            print(f'idx = {max_idx}')
            print(f'K_slow = {K_slow[row][col]}')
            print(f'C_slow_current = {C_slow_current[row][col]}')
            print(f'C_slow_past = {C_slow_past[row][col]}')
            print(f'dep_soc_fast = {dep_soc_fast[row][col]}')
            print(f'dep_soc_slow = {dep_soc_slow[row][col]}')
            print(f'ero_soc = {ero_soc[row][col] * (C_slow_past[row][col])}')
            print(f'A = {A[row][col]}')
            print(f'V = {V[row][col] * (1 - V_FAST_PROP)}')
            if past:
                print(f'L_slow = {L_slow[row][col]}')
            print(f'humification = {ALPHA * K_slow[row][col] * C_slow_past[row][col]}')
        else:
            max_idx = np.unravel_index(np.nanargmax(C_fast_current), C_fast_current.shape)
            row = max_idx[0]
            col = max_idx[1]
            print(f'idx = {max_idx}')
            print(f'K_fast = {K_fast[row][col]}')
            print(f'C_fast_current = {C_fast_current[row][col]}')
            print(f'dep_soc_fast = {dep_soc_fast[row][col]}')
            print(f'ero_soc = {cur_fast_proportion * ero_soc[row][col]}')
            print(f'A = {A[row][col]}')
            print(f'V = {V_FAST_PROP * V[row][col]}')
            print(f'humification = {ALPHA * K_fast[row][col] * C_fast_current[row][col]}')
            print('-----------------------------------------------------------------------')
            min_idx = np.unravel_index(np.nanargmax(C_slow_current), C_slow_current.shape)
            row = min_idx[0]
            col = min_idx[1]
            print(f'idx = {min_idx}')
            print(f'K_slow = {K_slow[row][col]}')
            print(f'C_slow_current = {C_slow_current[row][col]}')
            print(f'dep_soc_slow = {dep_soc_slow[row][col]}')
            print(f'ero_soc = {cur_slow_proportion * ero_soc[row][col]}')
            print(f'A = {A[row][col]}')
            print(f'V = {(1 - V_FAST_PROP) * V[row][col]}')
            print(f'humification = {ALPHA * K_slow[row][col] * C_slow_current[row][col]}')
        print('-----------------------------------------------------------------------')
        
    if PRINT_ALL:
        print(f'avg fast_proportion = {np.nanmean(C_fast_current / (C_slow_current + C_fast_current + 1e-9))}, max = {np.nanmax(C_fast_current / (C_slow_current + C_fast_current + 1e-9))}, min = {np.nanmin(C_fast_current / (C_slow_current + C_fast_current + 1e-9))}')
        print(f'avg K_fast = {np.nanmean(K_fast)}, max = {np.nanmax(K_fast)}, min = {np.nanmin(K_fast)}')
        print(f'avg K_slow = {np.nanmean(K_slow)}, max = {np.nanmax(K_slow)}, min = {np.nanmin(K_slow)}')
        print(f'avg C_fast_current = {np.nanmean(C_fast_current)}, max = {np.nanmax(C_fast_current)}, min = {np.nanmin(C_fast_current)}')
        print(f'avg C_slow_current = {np.nanmean(C_slow_current)}, max = {np.nanmax(C_slow_current)}, min = {np.nanmin(C_slow_current)}')
        print(f'avg dep_soc_fast = {np.nanmean(dep_soc_fast)}, max = {np.nanmax(dep_soc_fast)}, min = {np.nanmin(dep_soc_fast)}')
        print(f'avg dep_soc_slow = {np.nanmean(dep_soc_slow)}, max = {np.nanmax(dep_soc_slow)}, min = {np.nanmin(dep_soc_slow)}')
        if past:
            print(f'avg ero_soc = {np.nanmean(ero_soc * (C_fast_current + C_slow_current))}, max = {np.nanmax(ero_soc * (C_fast_current + C_slow_current))}, min = {np.nanmin(ero_soc * (C_fast_current + C_slow_current))}')
        else:
            print(f'avg ero_soc = {np.nanmean(ero_soc)}, max = {np.nanmax(ero_soc)}, min = {np.nanmin(ero_soc)}')
        print(f'avg A = {np.nanmean(A)}, max = {np.nanmax(A)}, min = {np.nanmin(A)}')
        print(f'avg V = {np.nanmean(V)}, max = {np.nanmax(V)}, min = {np.nanmin(V)}')
        if past:
            print(f'avg L_fast = {np.nanmean(L_fast)}, max = {np.nanmax(L_fast)}, min = {np.nanmin(L_fast)}')
        print(f'avg humification = {np.nanmean(ALPHA * K_fast * C_fast_current)}, max = {np.nanmax(ALPHA * K_fast * C_fast_current)}, min = {np.nanmin(ALPHA * K_fast * C_fast_current)}')
    # =============================================================================

    # Update SOC Pools
    if not past:
        C_fast_new = np.clip((C_fast_current + del_soc_fast), C_MIN_CAP, C_FAST_MAX)
        C_slow_new = np.clip((C_slow_current + del_soc_slow), C_MIN_CAP, C_SLOW_MAX)
    else:
        C_fast_new = np.clip(C_fast_past, C_MIN_CAP, C_FAST_MAX)
        C_slow_new = np.clip(C_slow_past, C_MIN_CAP, C_SLOW_MAX)
    
    MAP_STATS.C_fast_prev = C_fast_current.copy()
    MAP_STATS.C_slow_prev = C_slow_current.copy()
    MAP_STATS.C_fast_current = C_fast_new.copy()
    MAP_STATS.C_slow_current = C_slow_new.copy()
    
    return dep_soc_fast, dep_soc_slow, lost_soc

def get_deposition_of_point(E_tcell, A, point, dep_soil, dep_soc_fast, dep_soc_slow,
                            DEM, C_fast, C_slow, low_point_cur_stored, low_point_capacity,
                            small_boundary_mask, small_outlet_mask, large_boundary_mask,
                            large_outlet_mask, border_mask,):
    row, col = point
    total_slope = 0.0
    neighbours = []  # neighbours indicies that are lower than cur point
    nrows, ncols = DEM.shape
    for (dx, dy), dist in zip(_NEIGHBOR_OFFSETS, _NEIGHBOR_DIST):
        nr, nc = row + dx, col + dy
        # map bounds check
        if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
            continue
        # Skip if neighbour is outside the Loess Plateau boundary
        if not border_mask[nr][nc]:
            continue
        # Check Boundaries
        if small_boundary_mask[row, col] != small_boundary_mask[nr,nc] and not small_outlet_mask[row, col]:
            continue
        if large_boundary_mask[row, col] != large_boundary_mask[nr,nc] and not large_outlet_mask[row, col]:
            continue
        if low_point_capacity[nr][nc] > 0 and low_point_cur_stored[nr][nc] >= low_point_capacity[nr][nc]:
            continue
        
        diff = DEM[row, col] - DEM[nr, nc]
        if diff > 0.0:
            slope = diff / dist
            neighbours.append((nr, nc, slope))
            total_slope += slope
            
    if total_slope == 0.0:
        return
    
    for nr, nc, slope in neighbours:
        w = slope / total_slope
        dep_soc_fast[nr, nc]  += A[row, col] * C_fast * w
        dep_soc_slow[nr, nc]  += A[row, col] * C_slow * w
        dep_soil[nr, nc] += E_tcell[row, col] * w