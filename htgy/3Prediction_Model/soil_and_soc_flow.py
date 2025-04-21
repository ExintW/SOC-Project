from globalss import *
from globals import *
from numba import njit, prange
import numpy as np

@njit(parallel=False)
def distribute_soil_and_soc_with_dams_numba(
        E_tcell, S, DEM, dam_capacity_arr, grid_x, grid_y,
        small_boundary_mask, small_outlet_mask,
        large_boundary_mask, large_outlet_mask,
        river_mask, sorted_indices,
        reverse=False
):
    """
    Routes soil & SOC downhill, accounting for dams, boundary outflow, and river loss.
    E_tcell: soil loss from each cell (t/cell/month)
    S: SOC mass eroded from each cell (kg/cell/month)
    dam_capacity_arr: per-cell dam capacity (tons)
    DEM: digital elevation model
    river_mask: if a neighbor is a river, that portion of SOC is 'lost'
    boundary_masks & outlet_masks: control cross‐boundary flow
    """
    rows, cols = DEM.shape

    # accumulators
    inflow_soil = np.zeros((rows, cols), dtype=np.float64)
    inflow_soc  = np.zeros((rows, cols), dtype=np.float64)
    D_soil      = np.zeros((rows, cols), dtype=np.float64)
    D_soc       = np.zeros((rows, cols), dtype=np.float64)
    lost_soc    = np.zeros((rows, cols), dtype=np.float64)

    total_cells = sorted_indices.shape[0]

    def local_atomic_add(arr, idx, value):
        arr[idx[0], idx[1]] += value

    for index in range(total_cells):
        # 选定格点：正向 or 反向
        idx = index if not reverse else total_cells - 1 - index
        i, j = sorted_indices[idx, 0], sorted_indices[idx, 1]

        # 上游流入
        dep_soil = inflow_soil[i, j]
        dep_soc  = inflow_soc[i, j]

        # —— 大坝截留逻辑（保持不变） ——
        if dam_capacity_arr[i, j] > 0.0:
            cap = dam_capacity_arr[i, j]
            if reverse:
                take_soil = min(cap, cap)
                D_soil[i, j] = -take_soil
                fraction      = take_soil / cap if cap > 0 else 0.0
                D_soc[i, j]   = -dep_soc * fraction if dep_soc > 0 else 0.0
                dam_capacity_arr[i, j] = cap - take_soil
                dep_soil += take_soil
                dep_soc  += -D_soc[i, j]
            else:
                if dep_soil <= cap:
                    D_soil[i, j] = dep_soil
                    D_soc[i, j]  = dep_soc
                    dam_capacity_arr[i, j] = cap - dep_soil
                    dep_soil, dep_soc = 0.0, 0.0
                else:
                    D_soil[i, j] = cap
                    frac0 = cap / dep_soil if dep_soil > 0 else 0.0
                    D_soc[i, j]  = dep_soc * frac0
                    dam_capacity_arr[i, j] = 0.0
                    dep_soil -= cap
                    dep_soc  -= D_soc[i, j]
        else:
            # 无剩余容量，全部通行
            D_soil[i, j] = dep_soil
            D_soc[i, j]  = dep_soc

        # —— 本地侵蚀/输入源 ——
        if reverse:
            source_soil = -E_tcell[i, j]
            source_soc  = -S[i, j]
        else:
            source_soil =  E_tcell[i, j]
            source_soc  =  S[i, j]

        # 总量 = 流入 + 本地来源
        current_soil = dep_soil + source_soil
        current_soc  = dep_soc  + source_soc

        # —— 找到所有下游邻居并累加坡度 ——
        total_slope   = 0.0
        neighbor_count = 0
        neighbor_indices = np.empty((8, 2), dtype=np.int64)
        slope_diffs      = np.empty(8,    dtype=np.float64)

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                # 边界检查
                if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    continue

                elev_diff = DEM[i, j] - DEM[ni, nj]
                # 保证真降坡（正向）/真上坡（反向）
                if (not reverse and elev_diff <= 0) or (reverse and elev_diff >= 0):
                    continue

                # 计算坡度
                dist  = np.hypot(di, dj) + 1e-9
                slope = abs(elev_diff) / dist

                # 边界出口检查
                if small_boundary_mask[i,j] != small_boundary_mask[ni,nj] and not small_outlet_mask[i,j]:
                    continue
                if large_boundary_mask[i,j] != large_boundary_mask[ni,nj] and not large_outlet_mask[i,j]:
                    continue

                # 记录邻居及坡度
                neighbor_indices[neighbor_count, 0] = ni
                neighbor_indices[neighbor_count, 1] = nj
                slope_diffs[neighbor_count]        = slope
                total_slope += slope
                neighbor_count += 1

        # —— 按比例分配 ——
        if total_slope > 0.0:
            for k in range(neighbor_count):
                ni, nj = neighbor_indices[k]
                slope   = slope_diffs[k]
                frac    = slope / total_slope

                if river_mask[ni, nj] and not reverse:
                    # 只有流向河流的那部分 SOC 会丢失
                    local_atomic_add(lost_soc, (i, j), current_soc * frac)
                else:
                    # 其余方向按比例流入下游格点
                    local_atomic_add(inflow_soil, (ni, nj), source_soil * frac)
                    local_atomic_add(inflow_soc,  (ni, nj), source_soc  * frac)

    return D_soil, D_soc, inflow_soil, inflow_soc, lost_soc
