from utils import plot_SOC
from globalss import *
from globals import *  
import math
import time
from numba import njit
import torch
from UNet_Model import UNet

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
    return 0.1587 * np.log(LAI_safe) + 0.1331

def get_deposition_of_point(E_tcell, A, point, dep_soil, dep_soc_fast, dep_soc_slow,
                            DEM, C_fast, C_slow, low_point_cur_stored, low_point_capacity,
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
      

def soc_dynamic_model(E_tcell, A, sorted_indices, dam_max_cap, dam_cur_stored, active_dams, full_dams, V, month, year, past=False, UNet_MODEL=None, LAI_avg=None):
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
    dam_rem_cap = np.zeros(INIT_VALUES.DEM.shape, dtype=int)
    K_fast = INIT_VALUES.K_fast
    K_slow = INIT_VALUES.K_slow
    DEM = INIT_VALUES.DEM
    small_boundary_mask = MAP_STATS.small_boundary_mask
    small_outlet_mask = MAP_STATS.small_outlet_mask
    large_boundary_mask = MAP_STATS.large_boundary_mask
    large_outlet_mask = MAP_STATS.large_outlet_mask
    loess_border_mask = MAP_STATS.loess_border_mask
    
    if past and USE_UNET:
        # 构造输入张量 [1, C, H, W]
        # 动态变量：soc_fast、soc_slow、v_fast、v_slow、precip、check_dams
        # 静态变量：dem、loess_border_mask、river_mask、小边界、大边界、小出口、大出口
        V_fast = V_FAST_PROP * V
        V_slow = (1 - V_FAST_PROP) * V
        cf = np.nan_to_num(MAP_STATS.C_fast_current, nan=0.0)
        cs = np.nan_to_num(MAP_STATS.C_slow_current, nan=0.0)
        vf = np.nan_to_num(V_fast, nan=0.0)
        vs = np.nan_to_num(V_slow, nan=0.0)
        a = np.nan_to_num(A, nan=0.0)
        dem = np.nan_to_num(INIT_VALUES.DEM, nan=0.0)
        x = np.stack([
            cf,                # fast pool
            cs,                # slow pool
            vf,                 # v_fast
            vs,                 # v_slow 或替换为实际 v_slow
            a,                 # precip
            active_dams,       # check_dams，如有数据请替换
        ], axis=0)
        x = np.concatenate([
            x,
            dem[None],
            MAP_STATS.loess_border_mask[None],
            MAP_STATS.river_mask[None],
            MAP_STATS.small_boundary_mask[None],
            MAP_STATS.large_boundary_mask[None],
            MAP_STATS.small_outlet_mask[None],
            MAP_STATS.large_outlet_mask[None]
        ], axis=0)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = UNet_MODEL(x_tensor)
        pred_np = pred.squeeze(0).cpu().numpy()
        C_fast_new, C_slow_new = pred_np[0], pred_np[1]
        C_fast_new[~MAP_STATS.loess_border_mask] = np.nan
        C_slow_new[~MAP_STATS.loess_border_mask] = np.nan
        return C_fast_new, C_slow_new, np.zeros(DEM.shape, np.float64), np.zeros(DEM.shape, np.float64), np.zeros(DEM.shape, np.float64)

    init_fast_proportion = MAP_STATS.p_fast_grid
    init_slow_proportion = 1 - MAP_STATS.p_fast_grid
    
    soc_prev_fast = MAP_STATS.C_fast_prev
    soc_prev_slow = MAP_STATS.C_slow_prev

    shape = DEM.shape
    dep_soil = np.zeros(shape, np.float64)
    dep_soc_fast   = np.zeros(shape, np.float64)
    dep_soc_slow   = np.zeros(shape, np.float64)
    ero_soil  = np.zeros(shape, np.float64)
    ero_soc   = np.zeros(shape, np.float64)
    lost_soc = np.zeros(shape, dtype=np.float64)      # keep track of river losses
    
    max_A = np.nanmax(A)
    max_k_fast = np.nanmax(K_fast)
    max_k_slow = np.nanmax(K_slow)
    MAX_V = np.nanmax(V)
    
    if USE_1980_EQUIL and past:
        soc_1980_fast = INIT_VALUES.SOC_1980_FAST
        soc_1980_slow = INIT_VALUES.SOC_1980_SLOW
    if RUN_FROM_EQUIL and past:
        # if abs(year - 1980) > abs(year - EQUIL_YEAR) and year != EQUIL_YEAR:
        #     MAP_STATS.C_fast_equil_list *= EQUIL_DECREASE_FACTOR
        #     MAP_STATS.C_slow_equil_list *= EQUIL_DECREASE_FACTOR
        soc_equil_fast = MAP_STATS.C_fast_equil_list[month]
        soc_equil_slow = MAP_STATS.C_slow_equil_list[month]
        
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
    
    C_equil_fast = np.zeros(shape, np.float64)
    C_equil_slow = np.zeros(shape, np.float64)

    A = np.clip(A, 0, A_MAX)
    
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
        K_slow[row][col] = min(K_slow[row][col], K_SLOW_MAX)
        # V[row][col] = min(V[row][col] * 100, 0.4)
        # dep_soc[row][col] = min(dep_soc[row][col], D_MAX)
        V[row][col] += V_SCALING_FACTOR * (C_fast_current[row][col] + C_slow_current[row][col])
        
        if river_mask[row][col]:
            lost_soc[row][col] += dep_soc_fast[row][col] + dep_soc_slow[row][col]
            C_fast_current[row][col] = 0.0
            C_slow_current[row][col] = 0.0
            continue

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
            # L_fast[row][col] -= ero_soc[row][col] * init_fast_proportion[row][col]
            # L_slow[row][col] -= ero_soc[row][col] * init_slow_proportion[row][col]
            L_fast[row][col] -= ero_soc[row][col]
            L_slow[row][col] -= ero_soc[row][col]
            
            L_fast[row][col] = max(L_fast[row][col], L_FAST_MIN)
            L_slow[row][col] = max(L_slow[row][col], L_SLOW_MIN)
            
            # C_fast_past[row][col] = C_fast_current[row][col] - (init_fast_proportion[row][col] * V[row][col])
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
                if USE_1980_EQUIL:
                    if USE_1980_EQUIL_AVG:
                        C_equil_fast[row][col] = (soc_1980_fast[row][col] + soc_equil_fast[row][col]) / 2
                        C_equil_slow[row][col] = (soc_1980_slow[row][col] + soc_equil_slow[row][col]) / 2
                    elif USE_1980_EQUIL_PREV_AVG:
                        C_equil_fast[row][col] = (soc_1980_fast[row][col] + soc_equil_fast[row][col] + soc_prev_fast[row][col]) / 3
                        C_equil_slow[row][col] = (soc_1980_slow[row][col] + soc_equil_slow[row][col] + soc_prev_slow[row][col]) / 3
                    elif USE_DYNAMIC_AVG and not ALWAYS_USE_1980:
                        if year < 1980:
                            # if less than 1980: use 1980 with LAI trend as prior
                            if USE_1980_LAI_TREND:
                                C_equil_fast[row][col] = soc_1980_fast[row][col] * (LAI_avg / INIT_VALUES.LAI_1980[month])
                                C_equil_slow[row][col] = soc_1980_slow[row][col] * (LAI_avg / INIT_VALUES.LAI_1980[month])
                            else:
                                C_equil_fast[row][col] = soc_1980_fast[row][col]
                                C_equil_slow[row][col] = soc_1980_slow[row][col]
                        else:
                            w_equil = (year - 1980) / (EQUIL_YEAR - 1980)
                            w_1980 = 1 - w_equil
                            C_equil_fast[row][col] = w_1980 * soc_1980_fast[row][col] + w_equil * soc_equil_fast[row][col]
                            C_equil_slow[row][col] = w_1980 * soc_1980_slow[row][col] + w_equil * soc_equil_slow[row][col]
                        
                    elif abs(year - 1980) < abs(year - EQUIL_YEAR) or ALWAYS_USE_1980:
                        C_equil_fast[row][col] = soc_1980_fast[row][col]
                        C_equil_slow[row][col] = soc_1980_slow[row][col]
                    else:
                        C_equil_fast[row][col] = soc_equil_fast[row][col]
                        C_equil_slow[row][col] = soc_equil_slow[row][col]
                    
                    if USE_1980_LAI_TREND and (abs(year - 1980) < abs(year - EQUIL_YEAR) or ALWAYS_USE_1980) and not USE_DYNAMIC_AVG:
                        C_equil_fast[row][col] = soc_1980_fast[row][col] * (LAI_avg / INIT_VALUES.LAI_1980[month])
                        C_equil_slow[row][col] = soc_1980_slow[row][col] * (LAI_avg / INIT_VALUES.LAI_1980[month])
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
            # del_soc_fast[row][col] += init_fast_proportion[row][col] * (dep_soc[row][col] - ero_soc[row][col] + V[row][col]) - (K_fast[row][col] * C_fast_current[row][col])
            # del_soc_slow[row][col] += init_slow_proportion[row][col] * (dep_soc[row][col] - ero_soc[row][col] + V[row][col]) - (K_slow[row][col] * C_slow_current[row][col])
            del_soc_fast[row][col] += dep_soc_fast[row][col] - cur_fast_proportion * ero_soc[row][col] - (K_fast[row][col] * C_fast_current[row][col]) + V_FAST_PROP * V[row][col]
            del_soc_slow[row][col] += dep_soc_slow[row][col] - cur_slow_proportion * ero_soc[row][col] - (K_slow[row][col] * C_slow_current[row][col]) + (1 - V_FAST_PROP) * V[row][col] + ALPHA * K_fast[row][col] * C_fast_current[row][col]
        
        if dam_proportion > 0:
            time1 = time.time()
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
                                    loess_border_mask)
            time2 = time.time()
            total_dep_time += time2 - time1
    
    if past and USE_TIKHONOV and MAP_STATS.REG_counter == 1:
        MAP_STATS.REG_counter = REG_FREQ
        print(f"C_equil_fast: avg = {np.nanmean(C_equil_fast)}")
        print(f"C_equil_slow: avg = {np.nanmean(C_equil_slow)}")
        if LAI_avg is not None:
            print(f"LAI Proportion: {LAI_avg / INIT_VALUES.LAI_1980[month]}")
            
        if PLOT_PRIOR:
            C_equil_total = C_equil_fast + C_equil_slow
            C_equil_total[~MAP_STATS.loess_border_mask] = np.nan
            plot_SOC(C_equil_total, year, month, ext='Prior')
                
    elif past and USE_TIKHONOV:
        MAP_STATS.REG_counter -= 1
        
    MAP_STATS.dam_cur_stored = dam_cur_stored    
    time_end = time.time()
    
    print_max = True
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
        
    print_all = True
    if print_all:
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
        # print(f'max diff = {np.nanmax(C_fast_current) - np.nanmax(C_fast_prev)}')
        # print(f'max Damp = {LAMBDA_FAST * (np.nanmax(C_fast_current) - np.nanmax(C_fast_prev))}')

    print(f"total time: {time_end - time_start}")
    print(f"dep time: {total_dep_time}")
    
    if not past:
        C_fast_new = np.clip((C_fast_current + del_soc_fast), C_MIN_CAP, C_FAST_MAX)
        C_slow_new = np.clip((C_slow_current + del_soc_slow), C_MIN_CAP, C_SLOW_MAX)
        # damp_fast = LAMBDA_FAST * (C_fast_new - C_fast_current)
        # damp_slow = LAMBDA_SLOW * (C_slow_new - C_slow_current)
        # C_fast_new -= damp_fast
        # C_slow_new -= damp_slow
    else:
        C_fast_new = np.clip(C_fast_past, C_MIN_CAP, C_FAST_MAX)
        C_slow_new = np.clip(C_slow_past, C_MIN_CAP, C_SLOW_MAX)
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
        
    MAP_STATS.C_fast_prev = C_fast_current.copy()
    MAP_STATS.C_slow_prev = C_slow_current.copy()
    
    return C_fast_new, C_slow_new, dep_soc_fast, dep_soc_slow, lost_soc, full_dams, dam_rem_cap, dam_cur_stored