import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize
import time
import matplotlib.pyplot as plt
from numba import njit, prange

from globalss import *
from globals import *  


"""
Important Debugging Notes:
--------------------------
 - If your shapefiles have different CRSes (e.g., EPSG:32649 for small/large boundary vs. EPSG:4326 for river),
   you must reproject them carefully to a common projected CRS before buffering.
 - Print bounding boxes before & after reprojecting to confirm they match the region you expect.
 - If 'mask_file' exists from a previous run, it won't re-rasterize. Delete it if you've changed the shapefiles
   or any buffering logic.
"""
def precompute_river_basin():
    dx = np.mean(np.diff(MAP_STATS.grid_x))
    dy = np.mean(np.diff(MAP_STATS.grid_y))
    resolution = np.mean([dx, dy])  # In degrees if MAP_STATS.grid_x, MAP_STATS.grid_y are lat/lon
    buffer_distance = resolution    # This is in degrees if you're in EPSG:4326.
                                    # For real buffering in meters, reproject to e.g. EPSG:3857 or UTM.

    # Create affine transform for rasterization.
    minx = MAP_STATS.grid_x[0] - dx / 2
    maxy = MAP_STATS.grid_y[0] + dy / 2  # MAP_STATS.grid_y[0] is the max lat
    transform = Affine.translation(minx, maxy) * Affine.scale(dx, -dy)
    out_shape = (len(MAP_STATS.grid_y), len(MAP_STATS.grid_x))

    mask_file = OUTPUT_DIR / "PrecomputedMasks.npz"

    print("=== DEBUG: Grid bounding box ===")
    print(f"Grid longitude range: {MAP_STATS.grid_x.min():.6f} to {MAP_STATS.grid_x.max():.6f}")
    print(f"Grid latitude range : {MAP_STATS.grid_y.min():.6f} to {MAP_STATS.grid_y.max():.6f}")
    print("================================\n")

    if mask_file.exists():
        print("Loading precomputed masks...")
        masks = np.load(mask_file)
        MAP_STATS.small_boundary_mask = masks["small_boundary_mask"]
        MAP_STATS.large_boundary_mask = masks["large_boundary_mask"]
        MAP_STATS.river_mask = masks["river_mask"]
    else:
        print("Precomputing masks...")
        small_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "htgy_River_Basin.shp")
        large_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "94_area.shp")
        river_shp = gpd.read_file(DATA_DIR / "China_River" / "ChinaRiver_main.shp")

        # Filter river shapefile to only include features within the Loess Plateau border, if desired:
        river_shp = river_shp.to_crs(desired_crs)
        river_shp = river_shp[river_shp.intersects(MAP_STATS.loess_border_geom)]

        print("Small boundary reported CRS:", small_boundary_shp.crs)
        print("Small boundary total_bounds:", small_boundary_shp.total_bounds)
        print("Large boundary reported CRS:", large_boundary_shp.crs)
        print("Large boundary total_bounds:", large_boundary_shp.total_bounds)
        print("River reported CRS:", river_shp.crs)
        print("River total_bounds:", river_shp.total_bounds)

        # Reproject shapefiles to a projected CRS (e.g., EPSG:3857) for buffering
        proj_crs = "EPSG:3857"
        small_boundary_proj = small_boundary_shp.to_crs(proj_crs)
        large_boundary_proj = large_boundary_shp.to_crs(proj_crs)
        river_proj = river_shp.to_crs(proj_crs)

        print("=== DEBUG: Reprojected shapefile bounds (EPSG:3857) ===")
        print("Small boundary bounds:", small_boundary_proj.total_bounds)
        print("Large boundary bounds:", large_boundary_proj.total_bounds)
        print("River bounds:", river_proj.total_bounds)
        print("======================================================\n")

        t0 = time.perf_counter()
        small_boundary_buffered_proj = small_boundary_proj.buffer(buffer_distance)
        large_boundary_buffered_proj = large_boundary_proj.buffer(buffer_distance)
        river_buffered_proj = river_proj.buffer(buffer_distance)
        print(f"Buffering completed in {time.perf_counter() - t0:.2f} seconds.")

        # Reproject buffered geometries back to desired_crs (EPSG:4326) for rasterization
        small_boundary_buffered_gs = gpd.GeoSeries(small_boundary_buffered_proj, crs=proj_crs).to_crs(desired_crs)
        large_boundary_buffered_gs = gpd.GeoSeries(large_boundary_buffered_proj, crs=proj_crs).to_crs(desired_crs)
        river_buffered_gs = gpd.GeoSeries(river_buffered_proj, crs=proj_crs).to_crs(desired_crs)

        print("=== DEBUG: Buffered boundaries (EPSG:4326) bounds ===")
        print("Small boundary final bounds:", small_boundary_buffered_gs.total_bounds)
        print("Large boundary final bounds:", large_boundary_buffered_gs.total_bounds)
        print("River final bounds:", river_buffered_gs.total_bounds)
        print("======================================================\n")

        # Rasterize the buffered geometries
        MAP_STATS.small_boundary_mask = rasterize(
            [(geom, 1) for geom in small_boundary_buffered_gs.geometry],
            out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True
        ).astype(bool)

        MAP_STATS.large_boundary_mask = rasterize(
            [(geom, 1) for geom in large_boundary_buffered_gs.geometry],
            out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True
        ).astype(bool)

        MAP_STATS.river_mask = np.zeros(out_shape, dtype=bool)
        for geom in river_buffered_gs.geometry:
            MAP_STATS.river_mask |= rasterize(
                [(geom, 1)], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True
            ).astype(bool)

        print("Precomputed river_mask count:", np.count_nonzero(MAP_STATS.river_mask))
        np.savez(mask_file,
                small_boundary_mask=MAP_STATS.small_boundary_mask,
                large_boundary_mask=MAP_STATS.large_boundary_mask,
                river_mask=MAP_STATS.river_mask)

    print("Precomputed masks ready.")
    
    MAP_STATS.small_outlet_mask = compute_outlet_mask(MAP_STATS.small_boundary_mask, INIT_VALUES.DEM)
    MAP_STATS.large_outlet_mask = compute_outlet_mask(MAP_STATS.large_boundary_mask, INIT_VALUES.DEM)
    
    # ---------------------------------------------------------------------
    # Debug: Visualize boundary and river masks over the grid
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    # By default, imshow with extent sets axis limits to [minx, maxx, miny, maxy].
    # If your shapefile extends beyond that, it won't appear.
    # For debugging, you can remove the 'extent' or set bigger limits.
    X, Y = np.meshgrid(MAP_STATS.grid_x, MAP_STATS.grid_y)
    ax.imshow(np.zeros_like(INIT_VALUES.DEM), cmap='gray',
            extent=[MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(), MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()],
            origin='upper')  # Make sure origin='upper' matches your array orientation

    # Overlays:
    ax.contour(X, Y, MAP_STATS.small_boundary_mask, levels=[0.5], colors='orange', linestyles='--', linewidths=1.5)
    ax.contour(X, Y, MAP_STATS.large_boundary_mask, levels=[0.5], colors='green', linestyles='-', linewidths=1.5)
    ax.contour(X, Y, MAP_STATS.river_mask, levels=[0.5], colors='red', linestyles='-', linewidths=1.5)

    ax.set_title("Boundary and River Masks Overlay")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()

# ---------------------------------------------------------------------
# Helper: Compute outlet mask for a boundary.
# ---------------------------------------------------------------------
def compute_outlet_mask(boundary_mask, DEM):
    """
    Identify the lowest DEM cell within a boundary (the 'outlet').
    Returns a boolean mask with True only at the outlet cell.
    """
    outlet_mask = np.zeros_like(boundary_mask, dtype=bool)
    indices = np.where(boundary_mask)
    if len(indices[0]) > 0:
        min_index = np.argmin(DEM[boundary_mask])
        outlet_i = indices[0][min_index]
        outlet_j = indices[1][min_index]
        outlet_mask[outlet_i, outlet_j] = True
    return outlet_mask

@njit(parallel=True)
def distribute_soil_and_soc_with_dams_numba(
        E_tcell, S, DEM, dam_capacity_arr, grid_x, grid_y,
        small_boundary_mask, small_outlet_mask,
        large_boundary_mask, large_outlet_mask,
        river_mask, sorted_indices
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

    for idx in prange(total_cells):
        i = sorted_indices[idx, 0]
        j = sorted_indices[idx, 1]

        # 'dep_soil' and 'dep_soc' are what's inflowed from upstream
        dep_soil = inflow_soil[i, j]
        dep_soc = inflow_soc[i, j]

        # If there's a dam with remaining capacity, deposit soil up to that capacity
        if dam_capacity_arr[i, j] > 0.0:
            cap = dam_capacity_arr[i, j]
            if dep_soil <= cap:
                D_soil[i, j] = dep_soil
                D_soc[i, j] = dep_soc
                dam_capacity_arr[i, j] = cap - dep_soil
                excess_soil = 0.0
                excess_soc = 0.0
            else:
                D_soil[i, j] = cap
                fraction_deposited = (cap / dep_soil) if dep_soil > 0.0 else 0.0
                deposited_soc = dep_soc * fraction_deposited
                if deposited_soc < 0.0:
                    deposited_soc = 0.0
                D_soc[i, j] = deposited_soc
                dam_capacity_arr[i, j] = 0.0
                excess_soil = dep_soil - cap
                excess_soc = dep_soc - deposited_soc
            current_inflow_soil = excess_soil
            current_inflow_soc = excess_soc
        else:
            # No capacity; everything just passes through
            D_soil[i, j] = dep_soil
            D_soc[i, j] = dep_soc
            current_inflow_soil = dep_soil
            current_inflow_soc = dep_soc

        # Source from local erosion
        source_soil = E_tcell[i, j]
        source_soc = S[i, j]

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
                if DEM[ni, nj] >= DEM[i, j]:
                    continue
                # If neighbor is a river cell, we consider that portion 'lost'
                # and do not route further from that neighbor
                if river_mask[ni, nj]:
                    dist = np.hypot(di, dj) + 1e-9
                    slope_diff = (DEM[i, j] - DEM[ni, nj]) / dist
                    if slope_diff < 0.0:
                        slope_diff = 0.0
                    local_atomic_add(lost_soc, (i, j), source_soc * slope_diff)
                    continue
                # Check boundary crossing for small/large basins
                if small_boundary_mask[i, j] != small_boundary_mask[ni, nj]:
                    if not small_outlet_mask[i, j]:
                        continue
                if large_boundary_mask[i, j] != large_boundary_mask[ni, nj]:
                    if not large_outlet_mask[i, j]:
                        continue

                dist = np.hypot(di, dj) + 1e-9
                slope_diff = (DEM[i, j] - DEM[ni, nj]) / dist
                if slope_diff < 0.0:
                    slope_diff = 0.0
                total_slope += slope_diff
                neighbor_indices[neighbor_count, 0] = ni
                neighbor_indices[neighbor_count, 1] = nj
                slope_diffs[neighbor_count] = slope_diff
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


