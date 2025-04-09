import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize
import time
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches
from numba import njit, prange

from globalss import *
from globals import *  

def precompute_river_basin_2():
    # 加载数据
    small_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "htgy_River_Basin.shp")
    large_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "94_area.shp")
    river_shp = gpd.read_file(DATA_DIR / "China_River" / "ChinaRiver_main.shp")
    loess_border = MAP_STATS.loess_border_geom

    # 坐标系统一为EPSG:4326
    desired_crs = "EPSG:4326"
    small_boundary_shp = small_boundary_shp.to_crs(desired_crs)
    large_boundary_shp = large_boundary_shp.to_crs(desired_crs)
    river_shp = river_shp.to_crs(desired_crs)

    # 裁剪矢量
    loess_border_gdf = gpd.GeoDataFrame(geometry=[loess_border], crs=desired_crs)
    small_boundary_clip = gpd.clip(small_boundary_shp, loess_border_gdf)
    large_boundary_clip = gpd.clip(large_boundary_shp, loess_border_gdf)
    river_clip = gpd.clip(river_shp, loess_border_gdf)

    # 创建掩膜
    dem = INIT_VALUES.DEM.copy()
    grid_x, grid_y = np.meshgrid(MAP_STATS.grid_x, MAP_STATS.grid_y)

    # 将边界转为Path
    border_poly = loess_border_gdf.geometry.iloc[0]
    border_coords = np.array(border_poly.exterior.coords)
    poly_path = MplPath(border_coords)

    # 判断每个点是否在多边形内
    points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    mask = poly_path.contains_points(points).reshape(dem.shape)

    # 将掩膜外区域设置为NaN
    dem_masked = np.where(mask, dem, np.nan)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制裁剪后的DEM底图
    ax.imshow(dem_masked, cmap='terrain', alpha=0.7,
              extent=[MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(),
                      MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()],
              origin='upper')

    # 黄土高原边界
    loess_border_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5, label='Loess Plateau Border')

    # 绘制裁剪后的小、大边界和河流
    small_boundary_clip.boundary.plot(ax=ax, edgecolor='grey', linestyle='--', linewidth=0.5, label='Small Boundary')
    large_boundary_clip.boundary.plot(ax=ax, edgecolor='green', linewidth=1.5, label='Large Boundary')
    river_clip.plot(ax=ax, color='blue', linewidth=1, label='Rivers')

    ax.set_title('Boundaries and Rivers within Loess Plateau (Masked DEM)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.legend()
    plt.grid(True)
    plt.show()

def precompute_river_basin_1(): # Working version with only plotting river as contours
    # === 创建 transform 和分辨率 ===
    dx = np.mean(np.diff(MAP_STATS.grid_x))
    dy = np.abs(np.mean(np.diff(MAP_STATS.grid_y)))  # 保证为正数
    transform = Affine.translation(MAP_STATS.grid_x[0], MAP_STATS.grid_y[0]) * Affine.scale(dx, -dy)
    out_shape = (len(MAP_STATS.grid_y), len(MAP_STATS.grid_x))
    loess_border = MAP_STATS.loess_border_geom
    desired_crs = "EPSG:4326"

    print("Raster extent:")
    print("  minx:", MAP_STATS.grid_x[0], "maxx:", MAP_STATS.grid_x[-1])
    print("  miny:", MAP_STATS.grid_y[-1], "maxy:", MAP_STATS.grid_y[0])
    print("Output shape:", out_shape)
    
    print("=== 加载矢量数据 ===")
    small_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "htgy_River_Basin.shp")
    large_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "94_area.shp")
    river_shp = gpd.read_file(DATA_DIR / "China_River" / "ChinaRiver_main.shp")

    small_boundary_shp = small_boundary_shp.to_crs(desired_crs)
    large_boundary_shp = large_boundary_shp.to_crs(desired_crs)
    river_shp = river_shp.to_crs(desired_crs)

    # Clip
    loess_border_gdf = gpd.GeoDataFrame(geometry=[loess_border], crs=desired_crs)
    small_boundary_clip = gpd.clip(small_boundary_shp, loess_border_gdf)   
    large_boundary_clip = gpd.clip(large_boundary_shp, loess_border_gdf)
    river_clip = gpd.clip(river_shp, loess_border_gdf)
    
    small_boundary_clip = gpd.GeoDataFrame(geometry=small_boundary_clip.intersection(loess_border), crs=desired_crs)
    large_boundary_clip = gpd.GeoDataFrame(geometry=large_boundary_clip.intersection(loess_border), crs=desired_crs)
    river_clip = gpd.GeoDataFrame(geometry=river_clip.intersection(loess_border), crs=desired_crs)

    print("Loess border total_bounds:", loess_border_gdf.total_bounds)
    print("Grid X range:", MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max())
    print("Grid Y range:", MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max())
    print("Small boundary bounds:", small_boundary_clip.total_bounds)
    print("Large boundary bounds:", large_boundary_clip.total_bounds)
    print("River bounds:", river_clip.total_bounds)

    # === 栅格化为掩膜 ===
    MAP_STATS.small_boundary_mask = rasterize(
        [(geom, 1) for geom in small_boundary_clip.geometry],
        out_shape=out_shape, transform=transform,
        fill=0, dtype=np.uint8, all_touched=True
    ).astype(bool)

    MAP_STATS.large_boundary_mask = rasterize(
        [(geom, 1) for geom in large_boundary_clip.geometry],
        out_shape=out_shape, transform=transform,
        fill=0, dtype=np.uint8, all_touched=True
    ).astype(bool)

    MAP_STATS.river_mask = rasterize(
        [(geom, 1) for geom in river_clip.geometry],
        out_shape=out_shape, transform=transform,
        fill=0, dtype=np.uint8, all_touched=True
    ).astype(bool)

    print("Masks computed:")
    print("  small_boundary_mask:", np.count_nonzero(MAP_STATS.small_boundary_mask), "pixels")
    print("  large_boundary_mask:", np.count_nonzero(MAP_STATS.large_boundary_mask), "pixels")
    print("  river_mask:", np.count_nonzero(MAP_STATS.river_mask), "pixels")

    MAP_STATS.small_outlet_mask = compute_outlet_mask(MAP_STATS.small_boundary_mask, INIT_VALUES.DEM)
    MAP_STATS.large_outlet_mask = compute_outlet_mask(MAP_STATS.large_boundary_mask, INIT_VALUES.DEM)
    
    # === Debug 绘图 ===
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(),
              MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()]

    # DEM底图
    dem = INIT_VALUES.DEM.copy()
    grid_x, grid_y = np.meshgrid(MAP_STATS.grid_x, MAP_STATS.grid_y)
    border_poly = loess_border_gdf.geometry.iloc[0]
    border_coords = np.array(border_poly.exterior.coords)
    poly_path = MplPath(border_coords)
    points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    mask = poly_path.contains_points(points).reshape(dem.shape)
    dem_masked = np.where(mask, dem, np.nan)

    ax.imshow(dem_masked, cmap='terrain', alpha=0.7,
              extent=[MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(),
                      MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()],
              origin='upper')
    
    X, Y = np.meshgrid(MAP_STATS.grid_x, MAP_STATS.grid_y)
    
    small_boundary_clip.boundary.plot(ax=ax, color='grey', linewidth=0.5, linestyle='--', label='Small Boundary')
    large_boundary_clip.boundary.plot(ax=ax, color='green', linewidth=1.5, label='Large Boundary')
    loess_border_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5, label='Loess Plateau Border')

    # 河流轮廓
    ax.contour(X, Y, MAP_STATS.river_mask, levels=[0.5], colors='blue', linewidths=1.0)
    ax.set_xlim(loess_border.bounds[0], loess_border.bounds[2])  # xmin, xmax
    ax.set_ylim(loess_border.bounds[1], loess_border.bounds[3])  # ymin, ymax
    ax.set_title("Boundary and River Masks")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



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
    buffer_distance = 50000 #resolution   # This is in degrees if you're in EPSG:4326.
                                    # For real buffering in meters, reproject to e.g. EPSG:3857 or UTM.
    river_buffer_meters = 20000
    print("buffer_distance (degree):", buffer_distance)
    print("dx, dy:", dx, dy)

    # Create affine transform for rasterization.
    minx = MAP_STATS.grid_x[0]
    maxy = MAP_STATS.grid_y[0] # MAP_STATS.grid_y[0] is the max lat
    transform = Affine.translation(minx, maxy) * Affine.scale(dx, -dy)
    
    # minx = MAP_STATS.grid_x[0] - dx / 2
    # maxy = MAP_STATS.grid_y[0] + dy / 2  # MAP_STATS.grid_y[0] is the max lat
    # transform = Affine.translation(minx, maxy) * Affine.scale(dx, -dy)
    out_shape = (len(MAP_STATS.grid_y), len(MAP_STATS.grid_x))
    print(f'out_shape = {out_shape}')
    
    print("Transform bounds:", minx, MAP_STATS.grid_x[-1] + dx / 2,
                         MAP_STATS.grid_y[-1] - dy / 2, maxy)
    
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
        river_shp = gpd.clip(river_shp, MAP_STATS.loess_border_geom)
        print("River line count after clip:", len(river_shp))
        # river_shp = river_shp[river_shp.intersects(MAP_STATS.loess_border_geom)]

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
        river_buffered_proj = river_proj.buffer(river_buffer_meters)
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
        MAP_STATS.river_mask = rasterize(
            [(geom, 1) for geom in river_buffered_gs.geometry],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        ).astype(bool)

        # for geom in river_buffered_gs.geometry:
        #     MAP_STATS.river_mask |= rasterize(
        #         [(geom, 1)], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True
        #     ).astype(bool)
            
        print("Precomputed river_mask count:", np.count_nonzero(MAP_STATS.river_mask))
        np.savez(mask_file,
                small_boundary_mask=MAP_STATS.small_boundary_mask,
                large_boundary_mask=MAP_STATS.large_boundary_mask,
                river_mask=MAP_STATS.river_mask)
        # plt.imshow(MAP_STATS.river_mask, origin='upper')
        # plt.title("River Mask Debug")
        # plt.colorbar()
        # plt.show()

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
    
    ax.imshow(INIT_VALUES.DEM, cmap='gray', alpha=0.5,
        extent=[MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(), MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()],
        origin='upper')  # Make sure origin='upper' matches your array orientation
    
    # ax.imshow(np.zeros_like(INIT_VALUES.DEM), cmap='gray', alpha=0.5,
    #         extent=[MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(), MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()],
    #         origin='upper')  # Make sure origin='upper' matches your array orientation
    
    # 绘制小边界（橙色虚线）
    ax.contour(X, Y, MAP_STATS.small_boundary_mask, levels=[0.5], colors='orange', linestyles='--', linewidths=1.5, origin='upper')

    # 绘制大边界（绿色实线）
    ax.contour(X, Y, MAP_STATS.large_boundary_mask, levels=[0.5], colors='green', linestyles='-', linewidths=1.5, origin='upper')

    # 绘制河流（红色实线）
    ax.contour(X, Y, MAP_STATS.river_mask, levels=[0.5], colors='red', linestyles='-', linewidths=1.5, origin='upper')

    # 手动构造图例元素
    legend_handles = [
        mpatches.Patch(color='orange', label='Small Boundary', linestyle='--'),
        mpatches.Patch(color='green', label='Large Boundary'),
        mpatches.Patch(color='red', label='River Mask')
    ]

    # 图例
    ax.legend(handles=legend_handles, loc='lower left', fontsize=10, frameon=True)

    ax.set_title("Boundary and River Masks Overlay", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.show()
    # # Overlays:
    # ax.contour(X, Y, MAP_STATS.small_boundary_mask, levels=[0.5], colors='orange', linestyles='--', linewidths=1.5)
    # ax.contour(X, Y, MAP_STATS.large_boundary_mask, levels=[0.5], colors='green', linestyles='-', linewidths=1.5)
    # ax.contour(X, Y, MAP_STATS.river_mask, levels=[0.5], colors='red', linestyles='-', linewidths=1.5)

    # ax.set_title("Boundary and River Masks Overlay")
    # ax.set_xlabel("Longitude")
    # ax.set_ylabel("Latitude")
    # plt.show()

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

