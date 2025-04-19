from globalss import *
from globals import *  
from numba import njit, prange

@njit(parallel=False)
def distribute_soil_and_soc_with_dams_numba(
        E_tcell, S, DEM, dam_capacity_arr, grid_x, grid_y,
        small_boundary_mask, small_outlet_mask,
        large_boundary_mask, large_outlet_mask,
        river_mask, sorted_indices,
        reverse=False
):
    """
    Routes soil & SOC downhill, accounting for dams and boundary outflow.
    E_tcell: soil loss from each cell (t/cell/month)
    S: SOC mass eroded from each cell (kg/cell/month)
    dam_capacity_arr: per-cell dam capacity (tons)
    DEM: digital elevation model
    river_mask: if a neighbor is a river, that portion of SOC is 'lost'
    boundary_mask: if crossing from inside to outside the boundary, only allowed at the 'outlet'
    """
    rows, cols = DEM.shape
    inflow_soil = np.zeros((rows, cols), dtype=np.float64)
    inflow_soc = np.zeros((rows, cols), dtype=np.float64)
    D_soil = np.zeros((rows, cols), dtype=np.float64)
    D_soc = np.zeros((rows, cols), dtype=np.float64)
    lost_soc = np.zeros((rows, cols), dtype=np.float64)
    total_cells = sorted_indices.shape[0]

    def local_atomic_add(arr, idx, value):
        arr[idx[0], idx[1]] += value

    
    for index in range(total_cells):
        idx = index if not reverse else total_cells - 1 - index
        i = sorted_indices[idx, 0]
        j = sorted_indices[idx, 1]

        # 'dep_soil' and 'dep_soc' are what's inflowed from upstream
        dep_soil = inflow_soil[i, j]
        dep_soc = inflow_soc[i, j]

        # If there's a dam with remaining capacity, deposit soil up to that capacity
        if dam_capacity_arr[i, j] > 0.0:
            cap = dam_capacity_arr[i, j]
            
            if reverse:
                take_soil = min(cap, cap)
                D_soil[i, j] = -take_soil          # 负号=从坝中取出
                fraction = take_soil / cap if cap > 0 else 0.0
                D_soc[i, j]  = -dep_soc * fraction if dep_soc>0 else 0.0
                dam_capacity_arr[i, j] = cap - take_soil
                dep_soil += take_soil
                dep_soc  += (-D_soc[i, j])
            else:
                if dep_soil <= cap:
                    D_soil[i, j] = dep_soil
                    D_soc[i, j] = dep_soc
                    dam_capacity_arr[i, j] = cap - dep_soil
                    dep_soil = 0.0
                    dep_soc = 0.0
                else:
                    D_soil[i, j] = cap
                    frac = cap/dep_soil if dep_soil>0 else 0.0
                    D_soc[i, j] = dep_soc*frac
                    dam_capacity_arr[i, j] = 0.0
                    dep_soil -= cap
                    dep_soc  -= D_soc[i, j]
        else:
            # No capacity; everything just passes through
            D_soil[i, j] = dep_soil
            D_soc[i, j] = dep_soc
            
        # Source from local erosion
        if reverse:
            source_soil = -E_tcell[i, j]  
            source_soc  = -S[i, j]
        else:
            source_soil =  E_tcell[i, j]
            source_soc  =  S[i, j]
            
        current_soil = dep_soil + source_soil
        current_soc  = dep_soc  + source_soc
            
        # We look for neighbors that are strictly lower in DEM
        total_slope = 0.0
        neighbor_count = 0
        neighbor_indices = np.empty((8, 2), dtype=np.int64)
        slope_diffs = np.empty(8, dtype=np.float64)

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = i + di
                nj = j + dj
                if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    continue
                
                elev_diff = DEM[i,j] - DEM[ni,nj]
                if (not reverse and elev_diff<=0) or (reverse and elev_diff>=0):
                    continue
                # If neighbor is a river cell, we consider that portion 'lost'
                # and do not route further from that neighbor
                if river_mask[ni, nj]:
                    if not reverse:
                        dist = np.hypot(di,dj)+1e-9
                        slope = elev_diff/dist if elev_diff>0 else 0.0
                        local_atomic_add(lost_soc, (i,j), current_soc * slope)
                    continue
                
                # Check boundary crossing for small/large basins
                if small_boundary_mask[i,j]!=small_boundary_mask[ni,nj] and not small_outlet_mask[i,j]:
                    continue
                if large_boundary_mask[i,j]!=large_boundary_mask[ni,nj] and not large_outlet_mask[i,j]:
                    continue
                
                dist = np.hypot(di, dj) + 1e-9
                slope = abs(elev_diff)/dist
                total_slope += slope
                neighbor_indices[neighbor_count,0] = ni
                neighbor_indices[neighbor_count,1] = nj
                slope_diffs[neighbor_count] = slope
                neighbor_count += 1

        # Distribute local soil & SOC to lower neighbors by slope fraction
        if total_slope > 0:
            for k in range(neighbor_count):
                ni = neighbor_indices[k, 0]
                nj = neighbor_indices[k, 1]
                fraction = slope_diffs[k] / total_slope
                local_atomic_add(inflow_soil, (ni, nj), source_soil * fraction)
                local_atomic_add(inflow_soc, (ni, nj), source_soc * fraction)

    return D_soil, D_soc, inflow_soil, inflow_soc, lost_soc


