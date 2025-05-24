from globalss import *
from globals import *  
import math
import time
from numba import njit

_NEIGHBOR_OFFSETS = ((-1, -1), (-1, 0), (-1, 1),
                    ( 0, -1),          ( 0, 1),
                    ( 1, -1), ( 1, 0), ( 1, 1))
_NEIGHBOR_DIST   = (math.sqrt(2), 1.0, math.sqrt(2),
                    1.0,                        1.0,
                    math.sqrt(2), 1.0, math.sqrt(2))

# =============================================================================
# VEGETATION INPUT & UPDATED SOC DYNAMIC MODEL
# =============================================================================

def vegetation_input(LAI):
    """
    Compute vegetation input based on LAI using an empirical formula.
    E.g., V = a * LAI + b
    # floor at a tiny positive value so log never sees zero or negatives
    """
    LAI_safe = np.maximum(LAI, 1e-6)
    return 0.11434652 * np.log(LAI_safe) + 0.08709953

# def soc_dynamic_model(C_fast, C_slow,
#                       soc_loss_g_kg_month, D_soil, D_soc, V,
#                       K_fast, K_slow, p_fast_grid, dt, M_soil, lost_soc):
#     """
#     Update SOC pools (g/kg) for one month.
#     - Erosion removes SOC (soc_loss_g_kg_month).
#     - Deposition adds SOC (converted from D_soc to g/kg).
#     - Vegetation adds new SOC input.
#     - Reaction (decay) reduces each pool at rates K_fast, K_slow.
#     - Lost SOC (e.g., to rivers) is subtracted.
#     """
#     # Erosion partitioned into fast & slow
#     erosion_fast = -soc_loss_g_kg_month * p_fast_grid
#     erosion_slow = -soc_loss_g_kg_month * (1 - p_fast_grid)

#     # Deposition: (D_soc * 1000) / M_soil -> convert t -> g, then per kg soil
#     deposition_concentration = (D_soc * 1000.0) / M_soil
#     deposition_fast = deposition_concentration * p_fast_grid
#     deposition_slow = deposition_concentration * (1 - p_fast_grid)

#     # Vegetation input
#     vegetation_fast = V * p_fast_grid
#     vegetation_slow = V * (1 - p_fast_grid)

#     # Reaction/decay
#     reaction_fast = -K_fast * C_fast
#     reaction_slow = -K_slow * C_slow

#     # Lost SOC partition
#     lost_fast = lost_soc * p_fast_grid
#     lost_slow = lost_soc * (1 - p_fast_grid)

#     # print(f"erosion_fast mean = {np.nanmean(erosion_fast)}")
#     # print(f"erosion_slow mean = {np.nanmean(erosion_slow)}")
#     # print(f"deposition_fast mean = {np.nanmean(deposition_fast)}")
#     # print(f"deposition_slow mean = {np.nanmean(deposition_slow)}")
#     # print(f"vegetation_fast mean = {np.nanmean(vegetation_fast)}")
#     # print(f"vegetation_slow mean = {np.nanmean(vegetation_slow)}")
#     # print(f"reaction_fast mean = {np.nanmean(reaction_fast)}")
#     # print(f"reaction_slow mean = {np.nanmean(reaction_slow)}")
#     # print(f"lost_fast mean = {np.nanmean(lost_fast)}")
#     # print(f"lost_slow mean = {np.nanmean(lost_slow)}")
    
#     # Update each pool
#     C_fast_new = np.maximum(
#         C_fast + (erosion_fast + deposition_fast + vegetation_fast + reaction_fast - lost_fast) * dt,
#         0
#     )
#     C_slow_new = np.maximum(
#         C_slow + (erosion_slow + deposition_slow + vegetation_slow + reaction_slow - lost_slow) * dt,
#         0
#     )
#     return C_fast_new, C_slow_new

# def soc_dynamic_model_past(C_fast, C_slow,
#                       soc_loss_g_kg_month, D_soil, D_soc, V,
#                       K_fast, K_slow, p_fast_grid, dt, M_soil, lost_soc):
#     """
#     Update SOC pools (g/kg) for one month.
#     - Erosion removes SOC (soc_loss_g_kg_month).
#     - Deposition adds SOC (converted from D_soc to g/kg).
#     - Vegetation adds new SOC input.
#     - Reaction (decay) reduces each pool at rates K_fast, K_slow.
#     - Lost SOC (e.g., to rivers) is subtracted.
#     """
#     # Erosion partitioned into fast & slow
#     erosion_fast = -soc_loss_g_kg_month * p_fast_grid
#     erosion_slow = -soc_loss_g_kg_month * (1 - p_fast_grid)

#     # Deposition: (D_soc * 1000) / M_soil -> convert t -> g, then per kg soil
#     deposition_concentration = (D_soc * 1000.0) / M_soil
#     deposition_fast = deposition_concentration * p_fast_grid
#     deposition_slow = deposition_concentration * (1 - p_fast_grid)

#     # Vegetation input
#     vegetation_fast = V * p_fast_grid
#     vegetation_slow = V * (1 - p_fast_grid)

#     # Reaction/decay
#     reaction_fast = C_fast - C_fast/(1-K_fast)
#     reaction_slow = C_slow - C_slow/(1-K_slow)

#     # Lost SOC partition
#     Lost_concentration = (lost_soc * 1000.0) / M_soil
#     lost_fast = Lost_concentration * p_fast_grid
#     lost_slow = Lost_concentration * (1 - p_fast_grid)

#     # Update each pool
#     C_fast_new = np.maximum(
#         C_fast + (erosion_fast - deposition_fast + vegetation_fast + reaction_fast - lost_fast) * dt,
#         0
#     )
#     C_slow_new = np.maximum(
#         C_slow + (erosion_slow - deposition_slow + vegetation_slow + reaction_slow - lost_slow) * dt,
#         0
#     )
#     return C_fast_new, C_slow_new

def get_deposition_of_point(E_tcell, A, point, dep_soil, dep_soc,
                            DEM, C_total, low_point_cur_stored, low_point_capacity,
                            small_boundary_mask, small_outlet_mask, large_boundary_mask,
                            large_outlet_mask, loess_border_mask,):
    row, col = point
    total_slope = 0.0
    neighbours = []  # neighbours indicies that are lower than cur point
    nrows, ncols = DEM.shape
    for (dx, dy), dist in zip(_NEIGHBOR_OFFSETS, _NEIGHBOR_DIST):
        nr, nc = row + dx, col + dy
        # map bounds check
        if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
            continue
        # 2) Skip if neighbour is outside the Loess Plateau boundary
        if not loess_border_mask[nr][nc]:
            continue
        # 流域检查
        if small_boundary_mask[row, col] != small_boundary_mask[nr,nc] and not small_outlet_mask[row, col]:
            continue
        if large_boundary_mask[row, col] != large_boundary_mask[nr,nc] and not large_outlet_mask[row, col]:
            continue
        if low_point_capacity[row][col] > 0 and low_point_cur_stored[row][col] >= low_point_capacity[row][col]:
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
        dep_soc[nr, nc]  += A[row, col] * C_total * w
        dep_soil[nr, nc] += E_tcell[row, col] * w
      

def soc_dynamic_model(E_tcell, A, sorted_indices, dam_max_cap, dam_cur_stored, active_dams, V, past=False):
    dt = 1
    if past:
        dt = -1
    V *= V_FACTOR
    V = np.clip(V, V_MIN_CLIP, None)
    C_fast_current = MAP_STATS.C_fast_current
    C_slow_current = MAP_STATS.C_slow_current
    river_mask = MAP_STATS.river_mask
    low_point_mask = MAP_STATS.low_mask
    low_point_capacity = MAP_STATS.Low_Point_Capacity
    low_point_cur_stored = np.zeros(INIT_VALUES.DEM.shape, dtype=np.float64)
    K_fast = INIT_VALUES.K_fast
    K_slow = INIT_VALUES.K_slow
    DEM = INIT_VALUES.DEM
    small_boundary_mask = MAP_STATS.small_boundary_mask
    small_outlet_mask = MAP_STATS.small_outlet_mask
    large_boundary_mask = MAP_STATS.large_boundary_mask
    large_outlet_mask = MAP_STATS.large_outlet_mask
    loess_border_mask = MAP_STATS.loess_border_mask

    init_fast_proportion = MAP_STATS.p_fast_grid
    init_slow_proportion = 1 - MAP_STATS.p_fast_grid

    shape = DEM.shape
    dep_soil = np.zeros(shape, np.float64)
    dep_soc   = np.zeros(shape, np.float64)
    ero_soil  = np.zeros(shape, np.float64)
    ero_soc   = np.zeros(shape, np.float64)
    lost_soc = np.zeros(shape, dtype=np.float64)      # keep track of river losses

    if not past:
        del_soc_fast  = np.zeros(shape, dtype=np.float64) # Change in SOC
        del_soc_slow  = np.zeros(shape, dtype=np.float64) # Change in SOC
        del_soc_fast[~MAP_STATS.loess_border_mask] = np.nan
        del_soc_slow[~MAP_STATS.loess_border_mask] = np.nan
    else:
        L_fast = np.zeros(shape, dtype=np.float64)
        L_slow = np.zeros(shape, dtype=np.float64)

        C_fast_past = np.zeros(shape, np.float64)
        C_slow_past = np.zeros(shape, np.float64)
        C_fast_past[~MAP_STATS.loess_border_mask] = np.nan
        C_slow_past[~MAP_STATS.loess_border_mask] = np.nan

    A = np.clip(A, None, A_MAX)
    
    total_dep_time = 0.0
    
    time_start = time.time()
    prev_dem = 1e9
    for point in sorted_indices:
        row = point[0]
        col = point[1]

        if not loess_border_mask[row][col]:
            continue
        
        assert DEM[row][col] <= prev_dem, f"Error: DEM not in decending order! cur_dem = {DEM[row][col]}, prev_dem = {prev_dem}"
        prev_dem= DEM[row][col]

        cur_fast_proportion = C_fast_current[row][col] / (C_slow_current[row][col] + C_fast_current[row][col] + 1e-9)
        cur_slow_proportion = C_slow_current[row][col] / (C_slow_current[row][col] + C_fast_current[row][col] + 1e-9)

        # A[row][col] = min(A[row][col], 0.1)
        # K_fast[row][col] = min(K_fast[row][col], 0.1)
        # K_slow[row][col] = min(K_slow[row][col], 0.1)
        # V[row][col] = min(V[row][col] * 100, 0.4)
        
        V[row][col] += V_SCALING_FACTOR * C_fast_current[row][col]
        
        if river_mask[row][col]:
            lost_soc[row][col] += dep_soc[row][col]
            C_fast_current[row][col] = 0.0
            C_slow_current[row][col] = 0.0
            continue

        if low_point_mask[row][col] and not active_dams[row][col]:
            ero_soc[row][col] = 0.0
            ero_soil[row][col] = 0.0
            if low_point_cur_stored[row][col] <= low_point_capacity[row][col]:
                low_point_cur_stored[row][col] += dep_soil[row][col]
            else:
                dep_soc[row][col] = 0.0
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
        
        if past:
            # L_fast[row][col] -= ero_soc[row][col] * init_fast_proportion[row][col]
            # L_slow[row][col] -= ero_soc[row][col] * init_slow_proportion[row][col]
            L_fast[row][col] -= ero_soc[row][col]
            L_slow[row][col] -= ero_soc[row][col]
            
            L_fast[row][col] = max(L_fast[row][col], L_FAST_MIN)
            
            # C_fast_past[row][col] = C_fast_current[row][col] - (init_fast_proportion[row][col] * V[row][col])
            C_fast_past[row][col] = C_fast_current[row][col] - (V_FAST_PROP * V[row][col])
            C_fast_past[row][col] /= L_fast[row][col] + 1e-9
            # C_slow_past[row][col] = C_slow_current[row][col] - (init_slow_proportion[row][col] * V[row][col])
            C_slow_past[row][col] = C_slow_current[row][col] - (1 - V_FAST_PROP) * V[row][col]
            C_slow_past[row][col] /= L_slow[row][col] + 1e-9

            if dep_soc[row][col] > 0.0:
                C_fast_past[row][col] -= (init_fast_proportion[row][col] * dep_soc[row][col]) / (L_fast[row][col] + 1e-9)
                C_slow_past[row][col] -= (init_slow_proportion[row][col] * dep_soc[row][col]) / (L_slow[row][col] + 1e-9)

            C_fast_past[row][col] = max(C_fast_past[row][col], 0)
            # for humification
            C_slow_past[row][col] -= (ALPHA * C_fast_past[row][col] * K_fast[row][col]) / (L_slow[row][col] + 1e-9)
            C_slow_past[row][col] = max(C_slow_past[row][col], 0)

        if not past:
            # del_soc_fast[row][col] += init_fast_proportion[row][col] * (dep_soc[row][col] - ero_soc[row][col] + V[row][col]) - (K_fast[row][col] * C_fast_current[row][col])
            # del_soc_slow[row][col] += init_slow_proportion[row][col] * (dep_soc[row][col] - ero_soc[row][col] + V[row][col]) - (K_slow[row][col] * C_slow_current[row][col])
            del_soc_fast[row][col] += init_fast_proportion[row][col] * dep_soc[row][col] - cur_fast_proportion * ero_soc[row][col] - (K_fast[row][col] * C_fast_current[row][col]) + V_FAST_PROP * V[row][col]
            del_soc_slow[row][col] += init_slow_proportion[row][col] * dep_soc[row][col] - cur_slow_proportion * ero_soc[row][col] - (K_slow[row][col] * C_slow_current[row][col]) + (1 - V_FAST_PROP) * V[row][col] + ALPHA * K_fast[row][col] * C_fast_current[row][col]
        
        if dam_proportion > 0:
            time1 = time.time()
            if past:
                C_total = C_fast_past[row][col] + C_slow_past[row][col]
            else:
                C_total = C_fast_current[row][col] + C_slow_current[row][col]
            get_deposition_of_point(E_tcell, A, point, dep_soil, dep_soc, DEM,
                                    C_total * dam_proportion, low_point_cur_stored, low_point_capacity,
                                    small_boundary_mask, small_outlet_mask,
                                    large_boundary_mask, large_outlet_mask,
                                    loess_border_mask)
            time2 = time.time()
            total_dep_time += time2 - time1
        
        
    MAP_STATS.dam_cur_stored = dam_cur_stored    
    time_end = time.time()
    
    print_max = False
    if print_max:
        if past:
            max_idx = np.unravel_index(np.nanargmax(C_fast_past), C_fast_past.shape)
            row = max_idx[0]
            col = max_idx[1]
            print(f'idx = {max_idx}')
            print(f'K_fast = {K_fast[row][col]}')
            print(f'C_fast_current = {C_fast_current[row][col]}')
            print(f'C_fast_past = {C_fast_past[row][col]}')
            print(f'K_slow = {K_slow[row][col]}')
            print(f'dep_soc = {dep_soc[row][col]}')
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
            print(f'dep_soc = {dep_soc[row][col]}')
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
            print(f'dep_soc = {init_fast_proportion[row][col] * dep_soc[row][col]}')
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
            print(f'dep_soc = {init_slow_proportion[row][col] * dep_soc[row][col]}')
            print(f'ero_soc = {cur_slow_proportion * ero_soc[row][col]}')
            print(f'A = {A[row][col]}')
            print(f'V = {(1 - V_FAST_PROP) * V[row][col]}')
            print(f'humification = {ALPHA * K_slow[row][col] * C_slow_current[row][col]}')
        print('-----------------------------------------------------------------------')
        
    print_all = True
    if print_all:
        print(f'avg fast_proportion = {np.nanmean(C_fast_current / (C_slow_current + C_fast_current + 1e-9))}, max = {np.nanmax(C_fast_current / (C_slow_current + C_fast_current + 1e-9))}, min = {np.nanmin(C_fast_current / (C_slow_current + C_fast_current + 1e-9))}')
        print(f'avg K_fast = {np.nanmean(K_fast)}, max = {np.nanmax(K_fast)}, min = {np.nanmin(K_fast)}')
        print(f'avg K_slow = {np.nanmean(K_slow)}, max = {np.nanmax(K_slow)}, min = {np.nanmin(K_slow)}')
        print(f'avg C_fast_current = {np.nanmean(C_fast_current)}, max = {np.nanmax(C_fast_current)}, min = {np.nanmin(C_fast_current)}')
        print(f'avg C_slow_current = {np.nanmean(C_slow_current)}, max = {np.nanmax(C_slow_current)}, min = {np.nanmin(C_slow_current)}')
        print(f'avg dep_soc = {np.nanmean(dep_soc)}, max = {np.nanmax(dep_soc)}, min = {np.nanmin(dep_soc)}')
        if past:
            print(f'avg ero_soc = {np.nanmean(ero_soc * (C_fast_current + C_slow_current))}, max = {np.nanmax(ero_soc * (C_fast_current + C_slow_current))}, min = {np.nanmin(ero_soc * (C_fast_current + C_slow_current))}')
        else:
            print(f'avg ero_soc = {np.nanmean(ero_soc)}, max = {np.nanmax(ero_soc)}, min = {np.nanmin(ero_soc)}')
        print(f'avg A = {np.nanmean(A)}, max = {np.nanmax(A)}, min = {np.nanmin(A)}')
        print(f'avg V = {np.nanmean(V)}, max = {np.nanmax(V)}, min = {np.nanmin(V)}')
        if past:
            print(f'avg L_fast = {np.nanmean(L_fast)}, max = {np.nanmax(L_fast)}, min = {np.nanmin(L_fast)}')
        print(f'avg humification = {np.nanmean(ALPHA * K_fast * C_fast_current)}, max = {np.nanmax(ALPHA * K_fast * C_fast_current)}, min = {np.nanmin(ALPHA * K_fast * C_fast_current)}')
        # print(f'max diff = {np.nanmax(C_fast_current) - np.nanmax(C_fast_prev)}')
        # print(f'max Damp = {LAMBDA_FAST * (np.nanmax(C_fast_current) - np.nanmax(C_fast_prev))}')

    print(f"total time: {time_end - time_start}")
    print(f"dep time: {total_dep_time}")
    
    MAP_STATS.C_fast_prev = C_fast_current.copy()
    MAP_STATS.C_slow_prev = C_slow_current.copy()
    
    if not past:
        C_fast_new = np.maximum((C_fast_current + del_soc_fast), C_MIN_CAP)
        C_slow_new = np.maximum((C_slow_current + del_soc_slow), C_MIN_CAP)
        # damp_fast = LAMBDA_FAST * (C_fast_new - C_fast_current)
        # damp_slow = LAMBDA_SLOW * (C_slow_new - C_slow_current)
        # C_fast_new -= damp_fast
        # C_slow_new -= damp_slow
    else:
        C_fast_new = np.maximum(C_fast_past, C_MIN_CAP)
        C_slow_new = np.maximum(C_slow_past, C_MIN_CAP)
        damp_slow = LAMBDA_SLOW * (C_slow_new - C_slow_current)
        
        if np.nanmax(C_fast_new) > FAST_DAMP_START:
            C_fast_diff = (C_fast_new - C_fast_current)
            damp_fast = C_fast_diff[C_fast_diff > FAST_DAMP_THRESH] * LAMBDA_FAST
            C_fast_new[C_fast_diff > FAST_DAMP_THRESH] -= damp_fast
        C_slow_new -= damp_slow

    try:
        print(f'avg damping in fast = {np.nanmean(damp_fast)}, max = {np.nanmax(damp_fast)}, min = {np.nanmin(damp_fast)}')
    except:
        print('no damping this month')
    
    return C_fast_new, C_slow_new, dep_soc, lost_soc

# def soc_dynamic_model_past(E_tcell, A, sorted_indices, dam_max_cap, dam_cur_stored, active_dams, V):
#     C_fast_current = MAP_STATS.C_fast_current
#     C_slow_current = MAP_STATS.C_slow_current
#     river_mask = MAP_STATS.river_mask
#     K_fast = INIT_VALUES.K_fast     # setting K_fast = K_slow
#     K_slow = INIT_VALUES.K_slow
#     DEM = INIT_VALUES.DEM
#     small_boundary_mask = MAP_STATS.small_boundary_mask
#     small_outlet_mask = MAP_STATS.small_outlet_mask
#     large_boundary_mask = MAP_STATS.large_boundary_mask
#     large_outlet_mask = MAP_STATS.large_outlet_mask
#     loess_border_mask = MAP_STATS.loess_border_mask

#     shape = DEM.shape
#     dep_soil = np.zeros(shape, np.float64)
#     dep_soc   = np.zeros(shape, np.float64)
#     ero_soil  = np.zeros(shape, np.float64)
#     ero_soc   = np.zeros(shape, np.float64)
#     lost_soc = np.zeros(shape, dtype=np.float64)      # keep track of river losses

#     L_fast = np.zeros(shape, dtype=np.float64)
#     L_slow = np.zeros(shape, dtype=np.float64)

#     C_fast_past = np.zeros(shape, np.float64)
#     C_slow_past = np.zeros(shape, np.float64)
#     C_fast_past[~MAP_STATS.loess_border_mask] = np.nan
#     C_slow_past[~MAP_STATS.loess_border_mask] = np.nan

#     # fast_proportion = MAP_STATS.p_fast_grid
#     # slow_proportion = 1 - MAP_STATS.p_fast_grid

#     total_dep_time = 0.0

#     time_start = time.time()
#     for point in sorted_indices:
#         row = point[0]
#         col = point[1]

#         if not loess_border_mask[row][col]:
#             continue

#         # A[row][col] = min(A[row][col], 0.1)
#         # K_fast[row][col] = min(K_fast[row][col] * 0.015, 0.1)
#         # K_slow[row][col] = min(K_slow[row][col] * 0.015, 0.1)
#         # V[row][col] = min(V[row][col] * 100, 0.4)

#         fast_proportion = C_fast_current[row][col] / (C_slow_current[row][col] + C_fast_current[row][col] + 1e-9)
#         slow_proportion = C_slow_current[row][col] / (C_slow_current[row][col] + C_fast_current[row][col] + 1e-9)

#         if river_mask[row][col]:
#             lost_soc[row][col] += dep_soc[row][col]
#             C_fast_current[row][col] = 0.0
#             C_slow_current[row][col] = 0.0
#             continue

#         L_fast = 1 - K_fast[row][col]
#         L_slow = 1 - K_slow[row][col]

#         ero_soc[row][col] = A[row][col].copy()

#         dam_proportion = 1

#         if active_dams[row][col]:    # is dam
#             if dam_cur_stored[row][col] <= dam_max_cap[row][col]:
#                 ero_soc[row][col] = 0.0
#                 ero_soil[row][col] = 0.0
#                 dam_proportion = 0.0
#             else:
#                 extra_soil = dam_cur_stored[row][col] - dam_max_cap[row][col]
#                 assert extra_soil > 0, f'Error: dam extra soil <= 0: dam_cur_stored[row][col] = {dam_cur_stored[row][col]}, dam_max_cap[row][col] = {dam_max_cap[row][col]}'
#                 ero_proportion = extra_soil / dam_cur_stored[row][col]
#                 assert ero_proportion <= 1, f'Error: extra soil > total soil in dam: extra_soil = {extra_soil}, dam_cur_stored[row][col] = {dam_cur_stored[row][col]}, dam_max_cap[row][col] = {dam_max_cap[row][col]}'
#                 ero_soc[row][col] = A[row][col] * ero_proportion    # Only extra soc (over capacity) will erode
#                 ero_soil[row][col] = ero_proportion * E_tcell[row][col]    # Only extra soil (over capacity) will erode
#                 dam_proportion = ero_proportion
#             dam_cur_stored[row][col] -= dep_soil[row][col]
#             dam_cur_stored[row][col] += ero_soil[row][col]

#         L_fast -= ero_soc[row][col] * fast_proportion
#         L_slow -= ero_soc[row][col] * slow_proportion

#         C_fast_past[row][col] = C_fast_current[row][col] - (fast_proportion * V[row][col])
#         C_fast_past[row][col] /= L_fast + 1e-9
#         C_slow_past[row][col] = C_slow_current[row][col] - (slow_proportion * V[row][col])
#         C_slow_past[row][col] /= L_slow + 1e-9

#         if dep_soc[row][col] > 0.0:
#             C_fast_past[row][col] -= (fast_proportion * dep_soc[row][col]) / (L_fast + 1e-9)
#             C_slow_past[row][col] -= (slow_proportion * dep_soc[row][col]) / (L_slow + 1e-9)

#         C_fast_past[row][col] = max(C_fast_past[row][col], 0)
#         C_slow_past[row][col] = max(C_slow_past[row][col], 0)

#         if dam_proportion > 0:
#             time1 = time.time()
#             get_deposition_of_point(E_tcell, A, point, dep_soil, dep_soc, DEM,
#                                     (C_fast_past[row][col] + C_slow_past[row][col]) * dam_proportion,
#                                     small_boundary_mask, small_outlet_mask,
#                                     large_boundary_mask, large_outlet_mask,
#                                     loess_border_mask)
#             time2 = time.time()
#             total_dep_time += time2 - time1

#         # if (C_fast_past[row][col] + C_slow_past[row][col]) - (C_fast_current[row][col] + C_slow_current[row][col]) > 30:
#         #     print(f"Change in SOC: {(C_fast_past[row][col] + C_slow_past[row][col]) - (C_fast_current[row][col] + C_slow_current[row][col])}")
#         #     print(f"\tA = {A[row][col]}")
#         #     print()
#         #     print(f"\tNew SOC fast: {(C_fast_past[row][col])}")
#         #     print(f"\tOld SOC fast: {(C_fast_current[row][col])}")
#         #     print(f"\tero_soc fast: {ero_soc[row][col] * fast_proportion}")
#         #     print(f"\tdep_soc fast: {dep_soc[row][col] * fast_proportion}")
#         #     print(f"\tV fast: {V[row][col] * fast_proportion}")
#         #     print(f"\tL_fast: {L_fast}")
#         #     print(f"\tK_fast: {K_fast[row][col]}")
#         #     print(f"\tA fast: {A[row][col] * fast_proportion}")
#         #     print(f"\tfast proportion: {fast_proportion}")
#         #     print()
#         #     print(f"\tNew SOC slow: {(C_slow_past[row][col])}")
#         #     print(f"\tOld SOC slow: {(C_slow_current[row][col])}")
#         #     print(f"\tero_soc slow: {ero_soc[row][col] * slow_proportion}")
#         #     print(f"\tdep_soc slow: {dep_soc[row][col] * slow_proportion}")
#         #     print(f"\tV slow: {V[row][col] * slow_proportion}")
#         #     print(f"\tL_slow: {L_slow}")
#         #     print(f"\tK_slow: {K_slow[row][col]}")
#         #     print(f"\tA slow: {A[row][col] * slow_proportion}")
#         #     print(f"\tslow proportion: {slow_proportion}")

#     time_end = time.time()

#     print(f'avg fast_proportion = {np.nanmean(C_fast_current / (C_slow_current + C_fast_current + 1e-9))}, max = {np.nanmax(C_fast_current / (C_slow_current + C_fast_current + 1e-9))}, min = {np.nanmin(C_fast_current / (C_slow_current + C_fast_current + 1e-9))}')

#     print(f"total time: {time_end - time_start}")
#     print(f"dep time: {total_dep_time}")

#     return np.maximum(C_fast_past, 0), np.maximum(C_slow_past, 0), dep_soc, lost_soc

###########################################################
#                Inlined and flattened version
###########################################################

# def soc_dynamic_model_new(    
#     E_tcell,
#     A,
#     sorted_indicies,
#     dam_max_cap,
#     dam_cur_stored,
#     active_dams,
#     V,
#     reverse=False
# ):
#     """Flattened SOC dynamic model with inlined deposition logic (bit-for-bit identical)."""
#     # Dimensions
#     rows, cols = INIT_VALUES.DEM.shape
#     size = rows * cols

#     # Flattened 1D views of inputs
#     E_flat       = E_tcell.ravel()
#     A_flat       = A.ravel()
#     V_flat       = V.ravel()
#     dam_max_flat = dam_max_cap.ravel()
#     dam_cur_flat = dam_cur_stored.ravel()
#     active_flat  = active_dams.ravel()
#     river_flat   = MAP_STATS.river_mask.ravel()
#     Cf_flat      = MAP_STATS.C_fast_current.ravel()
#     Cs_flat      = MAP_STATS.C_slow_current.ravel()
#     Kf_flat      = INIT_VALUES.K_fast.ravel()
#     Ks_flat      = INIT_VALUES.K_slow.ravel()

#     # Buffers for deposition, erosion, deltas, etc.
#     dep_soil_2d   = np.zeros((rows, cols), np.float64)
#     dep_soc_2d    = np.zeros((rows, cols), np.float64)
#     dep_soil_flat = dep_soil_2d.ravel()
#     dep_soc_flat  = dep_soc_2d.ravel()

#     ero_soil_flat = np.zeros(size, np.float64)
#     ero_soc_flat  = np.zeros(size, np.float64)
#     del_fast_flat = np.zeros(size, np.float64)
#     del_slow_flat = np.zeros(size, np.float64)
#     lost_soc_flat = np.zeros(size, np.float64)
#     fast_prop     = np.zeros(size, np.float64)
#     slow_prop     = np.zeros(size, np.float64)

#     # Sorted indices by descending DEM
#     sorted_flat = np.argsort(INIT_VALUES.DEM.ravel())[::-1]

#     # Cached locals for inlined deposition
#     DEM  = INIT_VALUES.DEM
#     Cf2d = MAP_STATS.C_fast_current
#     Cs2d = MAP_STATS.C_slow_current
#     nrows, ncols = rows, cols

#     # Main loop
#     for idx in sorted_flat:
#         row = idx // cols
#         col = idx % cols

#         # Compute fast/slow proportions
#         total_C = Cf_flat[idx] + Cs_flat[idx] + 1e-9
#         fast_prop[idx] = Cf_flat[idx] / total_C
#         slow_prop[idx] = Cs_flat[idx] / total_C

#         # === Inlined deposition logic ===
#         total_slope = 0.0
#         nbrs = []
#         for (dx, dy), dist in zip(_NEIGHBOR_OFFSETS, _NEIGHBOR_DIST):
#             nr, nc = row + dx, col + dy
#             if not (0 <= nr < nrows and 0 <= nc < ncols):
#                 continue
#             diff = DEM[row, col] - DEM[nr, nc]
#             if diff > 0.0:
#                 slope = diff / dist
#                 nbrs.append((nr, nc, slope))
#                 total_slope += slope
#         if total_slope > 0.0:
#             Csum = Cf2d[row, col] + Cs2d[row, col]
#             for nr, nc, slope in nbrs:
#                 w = slope / total_slope
#                 dep_soc_flat[nr * cols + nc]  += A[nr, nc] * Csum * w
#                 dep_soil_flat[nr * cols + nc] += E_tcell[nr, nc] * w
#         # === End inlined deposition logic ===

#         # River loss
#         if river_flat[idx]:
#             lost_soc_flat[idx] += dep_soc_flat[idx]
#             continue

#         # Base erosion
#         ero_soc_flat[idx] = A_flat[idx] * (Cf_flat[idx] + Cs_flat[idx])
#         if active_flat[idx]:
#             if dam_cur_flat[idx] <= dam_max_flat[idx]:
#                 ero_soc_flat[idx] = 0.0
#                 ero_soil_flat[idx] = 0.0
#             else:
#                 extra = dam_cur_flat[idx] - dam_max_flat[idx]
#                 prop  = extra / dam_cur_flat[idx]
#                 ero_soc_flat[idx]  = A_flat[idx] * prop * (Cf_flat[idx] + Cs_flat[idx])
#                 ero_soil_flat[idx] = prop * E_flat[idx]
#             dam_cur_flat[idx] += dep_soil_flat[idx]
#             dam_cur_flat[idx] -= ero_soil_flat[idx]

#         # SOC delta updates
#         del_fast_flat[idx] += (
#             fast_prop[idx] * (dep_soc_flat[idx] + V_flat[idx] - ero_soc_flat[idx])
#             - (Kf_flat[idx] * Cf_flat[idx])
#         )
#         del_slow_flat[idx] += (
#             slow_prop[idx] * (dep_soc_flat[idx] + V_flat[idx] - ero_soc_flat[idx])
#             - (Ks_flat[idx] * Cs_flat[idx])
#         )

#     # Reconstruct 2D C arrays
#     C_fast_new = np.maximum(Cf_flat + del_fast_flat, 0.0).reshape((rows, cols))
#     C_slow_new = np.maximum(Cs_flat + del_slow_flat, 0.0).reshape((rows, cols))
#     return C_fast_new, C_slow_new

###########################################################
#                   NUMBA version
###########################################################

# @njit(nopython=True)
# def _soc_model_numba(
#     E_tcell, A,
#     dam_max_cap, dam_cur_stored,
#     active_dams, V,
#     DEM, C_fast_cur, C_slow_cur,
#     river_mask, K_fast, K_slow
# ):
#     rows, cols = DEM.shape
#     size = rows * cols

#     # Flattened inputs
#     E_flat       = E_tcell.ravel()
#     A_flat       = A.ravel()
#     V_flat       = V.ravel()
#     dam_max_flat = dam_max_cap.ravel()
#     dam_cur_flat = dam_cur_stored.ravel()
#     active_flat  = active_dams.ravel()
#     Cf_flat      = C_fast_cur.ravel()
#     Cs_flat      = C_slow_cur.ravel()
#     river_flat   = river_mask.ravel()
#     Kf_flat      = K_fast.ravel()
#     Ks_flat      = K_slow.ravel()

#     # Buffers
#     dep_soil = np.zeros(size, np.float64)
#     dep_soc  = np.zeros(size, np.float64)
#     ero_soil = np.zeros(size, np.float64)
#     ero_soc  = np.zeros(size, np.float64)
#     del_fast = np.zeros(size, np.float64)
#     del_slow = np.zeros(size, np.float64)
#     lost_soc = np.zeros(size, np.float64)
#     fast_prop= np.zeros(size, np.float64)
#     slow_prop= np.zeros(size, np.float64)

#     # Sorted DEM indices (descending)
#     DEM_flat = DEM.ravel()
#     sorted_flat = np.argsort(DEM_flat)[::-1]

#     for it in range(sorted_flat.shape[0]):
#         idx = sorted_flat[it]
#         row = idx // cols
#         col = idx % cols

#         # Fast/slow proportions
#         totC = Cf_flat[idx] + Cs_flat[idx] + 1e-9
#         fast_prop[idx] = Cf_flat[idx] / totC
#         slow_prop[idx] = Cs_flat[idx] / totC

#         # Deposition: two-pass using fixed-size neighbors arrays
#         total_slope = 0.0
#         nbr_r = np.empty(8, np.int64)
#         nbr_c = np.empty(8, np.int64)
#         nbr_s = np.empty(8, np.float64)
#         count = 0
#         for j in range(8):
#             dx, dy = _OFFSETS[j]
#             nr = row + dx; nc = col + dy
#             if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
#                 continue
#             diff = DEM[row, col] - DEM[nr, nc]
#             if diff > 0.0:
#                 slope = diff / _DISTS[j]
#                 nbr_r[count] = nr
#                 nbr_c[count] = nc
#                 nbr_s[count] = slope
#                 total_slope += slope
#                 count += 1
#         if total_slope > 0.0:
#             Csum = C_fast_cur[row, col] + C_slow_cur[row, col]
#             for j in range(count):
#                 w = nbr_s[j] / total_slope
#                 nidx = nbr_r[j] * cols + nbr_c[j]
#                 dep_soc[nidx]  += A_flat[nidx] * Csum * w
#                 dep_soil[nidx] += E_flat[nidx] * w

#         # River loss
#         if river_flat[idx]:
#             lost_soc[idx] += dep_soc[idx]
#             continue

#         # Erosion and dam logic
#         base = A_flat[idx] * (Cf_flat[idx] + Cs_flat[idx])
#         if active_flat[idx]:
#             cur = dam_cur_flat[idx]
#             cap = dam_max_flat[idx]
#             if cur <= cap:
#                 e_soil = 0.0
#                 e_soc  = 0.0
#             else:
#                 extra = cur - cap
#                 p     = extra / cur
#                 e_soc  = A_flat[idx] * p * (Cf_flat[idx] + Cs_flat[idx])
#                 e_soil = p * E_flat[idx]
#             dam_cur_flat[idx] = cur + dep_soil[idx] - e_soil
#             ero_soil[idx] = e_soil
#             ero_soc[idx]  = e_soc
#         else:
#             ero_soc[idx] = base

#         # SOC change
#         del_fast[idx] += fast_prop[idx] * (dep_soc[idx] + V_flat[idx] - ero_soc[idx]) - Kf_flat[idx] * Cf_flat[idx]
#         del_slow[idx] += slow_prop[idx] * (dep_soc[idx] + V_flat[idx] - ero_soc[idx]) - Ks_flat[idx] * Cs_flat[idx]

#     # Reshape outputs
#     C_fast_new = np.maximum(Cf_flat + del_fast, 0.0).reshape((rows, cols))
#     C_slow_new = np.maximum(Cs_flat + del_slow, 0.0).reshape((rows, cols))
#     return C_fast_new, C_slow_new

# def soc_dynamic_model_new(
#     E_tcell,
#     A,
#     sorted_indicies,
#     dam_max_cap,
#     dam_cur_stored,
#     active_dams,
#     V,
#     reverse=False
# ):
#     """User‐facing wrapper that pulls globals and calls the jitted function."""
#     return _soc_model_numba(
#         E_tcell, A,
#         dam_max_cap, dam_cur_stored,
#         active_dams, V,
#         INIT_VALUES.DEM,
#         MAP_STATS.C_fast_current,
#         MAP_STATS.C_slow_current,
#         MAP_STATS.river_mask,
#         INIT_VALUES.K_fast,
#         INIT_VALUES.K_slow
#     )