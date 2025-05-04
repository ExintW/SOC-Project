import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches
from numba import njit, prange
import os

from globalss import *
from globals import *  

def precompute_river_basin_2():
    # === 创建 transform 和分辨率 ===
    dx = np.mean(np.diff(MAP_STATS.grid_x))
    dy = np.abs(np.mean(np.diff(MAP_STATS.grid_y)))  # 保证为正数
    resolution = np.mean([dx, dy])  # In degrees if MAP_STATS.grid_x, MAP_STATS.grid_y are lat/lon
    buffer_distance = 1

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

    # Buffer
    proj_crs = "EPSG:3857"

    loess_border_gdf = gpd.GeoDataFrame(geometry=[loess_border], crs=desired_crs).to_crs(proj_crs)
    small_boundary_proj = small_boundary_shp.to_crs(proj_crs)
    large_boundary_proj = large_boundary_shp.to_crs(proj_crs)

    small_boundary_buffered_proj = small_boundary_proj.buffer(buffer_distance)
    large_boundary_buffered_proj = large_boundary_proj.buffer(buffer_distance)

    small_intersect = small_boundary_buffered_proj.intersection(loess_border_gdf.unary_union)
    large_intersect = large_boundary_buffered_proj.intersection(loess_border_gdf.unary_union)
    small_valid = gpd.GeoDataFrame(geometry=small_intersect, crs=proj_crs)
    large_valid = gpd.GeoDataFrame(geometry=large_intersect, crs=proj_crs)
    small_valid = small_valid[~small_valid.geometry.is_empty & small_valid.geometry.is_valid]
    large_valid = large_valid[~large_valid.geometry.is_empty & large_valid.geometry.is_valid]

    small_boundary_clip = small_valid.to_crs(desired_crs)
    large_boundary_clip = large_valid.to_crs(desired_crs)

    # small_boundary_clip = gpd.GeoDataFrame(geometry=small_boundary_buffered_proj.intersection(loess_border_gdf), crs=proj_crs).to_crs(desired_crs)
    # large_boundary_clip = gpd.GeoDataFrame(geometry=large_boundary_buffered_proj.intersection(loess_border_gdf), crs=proj_crs).to_crs(desired_crs)
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

    # DEM底图
    dem = INIT_VALUES.DEM.copy()
    extent = [
        MAP_STATS.grid_x[0],
        MAP_STATS.grid_x[0] + dx * dem.shape[1],
        MAP_STATS.grid_y[0] - dy * dem.shape[0],
        MAP_STATS.grid_y[0]
    ]
    grid_x, grid_y = np.meshgrid(MAP_STATS.grid_x, MAP_STATS.grid_y)
    loess_border_4326 = gpd.GeoDataFrame(geometry=[loess_border], crs=desired_crs)
    border_poly = loess_border_4326.geometry.iloc[0]
    border_coords = np.array(border_poly.exterior.coords)
    poly_path = MplPath(border_coords)
    points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    mask = poly_path.contains_points(points).reshape(dem.shape)
    dem_masked = np.where(mask, dem, np.nan)
    print("Mask valid pixels:", np.count_nonzero(mask))
    print("DEM shape:", dem.shape)
    print("grid_x shape:", grid_x.shape)
    print("grid_y shape:", grid_y.shape)

    print("small intersection_clip bounds:", small_boundary_clip.total_bounds)
    print("large intersection_clip bounds:", large_boundary_clip.total_bounds)
    print("Grid extent:", MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(), MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max())


    ax.imshow(dem_masked, cmap='terrain', alpha=0.7,
              extent=extent,
              origin='upper')

    X, Y = np.meshgrid(MAP_STATS.grid_x, MAP_STATS.grid_y)

    ax.contour(X, Y, MAP_STATS.small_boundary_mask, levels=[0.5], colors='grey', linewidths=0.5)
    ax.contour(X, Y, MAP_STATS.large_boundary_mask, levels=[0.5], colors='green', linewidths=1.5)
    loess_border_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5, label='Loess Plateau Border')

    # 河流轮廓
    ax.contour(X, Y, MAP_STATS.river_mask, levels=[0.5], colors='blue', linewidths=1.0)

    ax.set_xlim(MAP_STATS.grid_x[0], MAP_STATS.grid_x[-1])  # xmin, xmax
    ax.set_ylim(MAP_STATS.grid_y[-1], MAP_STATS.grid_y[0])  # ymin, ymax
    ax.set_title("Boundary and River Masks")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def precompute_river_basin_1(cache_path=None):
    """
    Precompute (or load) river-basin masks and Loess Plateau border mask with unique IDs.
    Saves to / loads from a compressed .npz to skip heavy GIS work on subsequent runs.

    Parameters:
        cache_path (path-like, optional): path to .npz cache file.
            Defaults to PROCESSED_DIR / "precomputed_masks.npz".

    On success, sets:
        MAP_STATS.small_boundary_mask   : 2D np.int32 array of small-basin IDs (0 = background)
        MAP_STATS.large_boundary_mask   : 2D np.int32 array of large-basin IDs
        MAP_STATS.river_mask            : 2D np.bool_ mask of river locations
        MAP_STATS.loess_border_mask     : 2D np.bool_ mask of Loess Plateau border
        MAP_STATS.small_outlet_mask     : 2D np.bool_ mask of small-basin outlets
        MAP_STATS.large_outlet_mask     : 2D np.bool_ mask of large-basin outlets
    """
    # 1) Determine & prepare cache file
    BORDER_PATH = DATA_DIR / "Loess_Plateau_vector_border.shp"

    if cache_path is None:
        cache_path = PROCESSED_DIR / "precomputed_masks.npz"
    os.makedirs(cache_path.parent, exist_ok=True)

    # 2) Try loading from cache
    if cache_path.exists():
        data = np.load(cache_path)
        MAP_STATS.small_boundary_mask = data["small_boundary_mask"]
        MAP_STATS.large_boundary_mask = data["large_boundary_mask"]
        MAP_STATS.river_mask = data["river_mask"]
        MAP_STATS.loess_border_mask = data["loess_border_mask"]
        MAP_STATS.small_outlet_mask = data["small_outlet_mask"]
        MAP_STATS.large_outlet_mask = data["large_outlet_mask"]
        print(f"Loaded precomputed masks from {cache_path}")
        return

    # 3) Otherwise compute from scratch
    # --- 3.1) Build raster transform & shape ---
    dx = np.mean(np.diff(MAP_STATS.grid_x))
    dy = abs(np.mean(np.diff(MAP_STATS.grid_y)))
    transform = Affine.translation(MAP_STATS.grid_x[0], MAP_STATS.grid_y[0]) * Affine.scale(dx, -dy)
    out_shape = (len(MAP_STATS.grid_y), len(MAP_STATS.grid_x))
    loess_border = MAP_STATS.loess_border_geom
    desired_crs = "EPSG:4326"

    # Original debug printouts
    print("Raster extent:")
    print("  minx:", MAP_STATS.grid_x[0], "maxx:", MAP_STATS.grid_x[-1])
    print("  miny:", MAP_STATS.grid_y[-1], "maxy:", MAP_STATS.grid_y[0])
    print("Output shape:", out_shape)

    print("=== 加载矢量数据 ===")
    # Load vector shapefiles and reproject
    small_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "htgy_River_Basin.shp").to_crs(desired_crs)
    large_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "94_area.shp").to_crs(desired_crs)
    river_shp = gpd.read_file(DATA_DIR / "China_River" / "ChinaRiver_main.shp").to_crs(desired_crs)
    border_shp = gpd.read_file(BORDER_PATH).to_crs(desired_crs)

    # Clip & intersect to Loess Plateau border
    loess_border_gdf = gpd.GeoDataFrame(geometry=[loess_border], crs=desired_crs)
    small_clip = gpd.clip(small_boundary_shp, loess_border_gdf).intersection(loess_border)
    large_clip = gpd.clip(large_boundary_shp, loess_border_gdf).intersection(loess_border)
    river_clip = gpd.clip(river_shp, loess_border_gdf).intersection(loess_border)
    border_clip = gpd.clip(border_shp, loess_border_gdf).intersection(loess_border)

    # Debug bounds
    print("Loess border total_bounds:", loess_border_gdf.total_bounds)
    print("Grid X range:", MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max())
    print("Grid Y range:", MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max())
    print("Small boundary bounds:", small_clip.total_bounds)
    print("Large boundary bounds:", large_clip.total_bounds)
    print("River bounds:", river_clip.total_bounds)
    print("Loess border vector bounds:", border_clip.total_bounds)

    # Explode multipart geometries into individual parts
    small_clip = small_clip.explode(index_parts=True).reset_index(drop=True)
    large_clip = large_clip.explode(index_parts=True).reset_index(drop=True)
    river_clip = river_clip.explode(index_parts=True).reset_index(drop=True)
    border_clip = border_clip.explode(index_parts=True).reset_index(drop=True)

    # 3.2) Rasterize masks
    # Small basins: ID = idx+1, background = 0
    small_shapes = [(geom, idx+1) for idx, geom in enumerate(small_clip.geometry)]
    MAP_STATS.small_boundary_mask = rasterize(
        small_shapes, out_shape=out_shape, transform=transform,
        fill=0, dtype=np.int32, all_touched=True
    )

    # Large basins: same logic
    large_shapes = [(geom, idx+1) for idx, geom in enumerate(large_clip.geometry)]
    MAP_STATS.large_boundary_mask = rasterize(
        large_shapes, out_shape=out_shape, transform=transform,
        fill=0, dtype=np.int32, all_touched=True
    )

    # Rivers: boolean mask
    MAP_STATS.river_mask = rasterize(
        [(geom, 1) for geom in river_clip.geometry],
        out_shape=out_shape, transform=transform,
        fill=0, dtype=np.uint8, all_touched=True
    ).astype(bool)

    # Loess Plateau border: boolean mask
    MAP_STATS.loess_border_mask = rasterize(
        [(geom, 1) for geom in border_clip.geometry],
        out_shape=out_shape, transform=transform,
        fill=0, dtype=np.uint8, all_touched=True
    ).astype(bool)

    # Print pixel counts
    print("Masks computed:")
    print(f"  small_boundary_mask: {np.count_nonzero(MAP_STATS.small_boundary_mask)} pixels (ID>0)")
    print(f"  large_boundary_mask: {np.count_nonzero(MAP_STATS.large_boundary_mask)} pixels (ID>0)")
    print(f"  river_mask: {np.count_nonzero(MAP_STATS.river_mask)} pixels (True)")
    print(f"  loess_border_mask: {np.count_nonzero(MAP_STATS.loess_border_mask)} pixels (True)")

    # 3.3) Count unique basin IDs
    small_ids = np.unique(MAP_STATS.small_boundary_mask)
    small_ids = small_ids[small_ids != 0]
    print(f"Number of small basin IDs: {len(small_ids)}")
    large_ids = np.unique(MAP_STATS.large_boundary_mask)
    large_ids = large_ids[large_ids != 0]
    print(f"Number of large basin IDs: {len(large_ids)}")

    # 3.4) Compute outlet masks
    MAP_STATS.small_outlet_mask = compute_multi_outlet_mask(
        geometries=small_clip.geometry,
        DEM=INIT_VALUES.DEM,
        out_shape=out_shape,
        transform=transform
    )
    MAP_STATS.large_outlet_mask = compute_multi_outlet_mask(
        geometries=large_clip.geometry,
        DEM=INIT_VALUES.DEM,
        out_shape=out_shape,
        transform=transform
    )

    # Print outlet counts
    print("Outlet masks computed:")
    print(f"  small_outlet_mask: {np.count_nonzero(MAP_STATS.small_outlet_mask)} pixels (True)")
    print(f"  large_outlet_mask: {np.count_nonzero(MAP_STATS.large_outlet_mask)} pixels (True)")

    # 3.5) Optional debug plot (not shown by default)
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(),
              MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()]
    dem = INIT_VALUES.DEM.copy()
    grid_x, grid_y = np.meshgrid(MAP_STATS.grid_x, MAP_STATS.grid_y)
    border_coords = np.array(loess_border.exterior.coords)
    mask_array = MplPath(border_coords).contains_points(
        np.vstack((grid_x.flatten(), grid_y.flatten())).T
    ).reshape(dem.shape)
    dem_masked = np.where(mask_array, dem, np.nan)
    ax.imshow(dem_masked, cmap='terrain', alpha=0.7,
              extent=extent, origin='upper')
    small_clip.boundary.plot(ax=ax, color='grey', linewidth=0.5,
                              linestyle='--', label='Small Boundary')
    large_clip.boundary.plot(ax=ax, color='green', linewidth=1.5,
                              label='Large Boundary')
    loess_border_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5,
                                    label='Loess Plateau Border')
    ax.contour(grid_x, grid_y, MAP_STATS.river_mask,
               levels=[0.5], colors='blue', linewidths=1.0)
    ax.set_xlim(loess_border.bounds[0], loess_border.bounds[2])
    ax.set_ylim(loess_border.bounds[1], loess_border.bounds[3])
    ax.set_title("Boundary and River Masks")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.close(fig)

    # 4) Save all masks to cache
    np.savez_compressed(
        cache_path,
        small_boundary_mask=MAP_STATS.small_boundary_mask,
        large_boundary_mask=MAP_STATS.large_boundary_mask,
        river_mask=MAP_STATS.river_mask,
        loess_border_mask=MAP_STATS.loess_border_mask,
        small_outlet_mask=MAP_STATS.small_outlet_mask,
        large_outlet_mask=MAP_STATS.large_outlet_mask,
    )
    print(f"Computed and saved masks to {cache_path}")



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
def compute_multi_outlet_mask(geometries, DEM, out_shape, transform):
    """
    Compute an outlet for each individual polygon provided in the 'geometries' iterable.
    For each polygon (sub-basin), rasterize it, then within that temporary mask find the DEM cell with the
    lowest value. Mark that cell in the global outlet mask.

    Parameters:
      geometries - an iterable of shapely geometries (each representing one sub-basin)
      DEM        - 2D NumPy array of DEM data
      out_shape  - tuple (rows, cols) for the output raster mask
      transform  - Affine transform mapping raster indices to coordinate space

    Returns:
      multi_outlet_mask - boolean array of shape out_shape with True at each computed outlet.
    """
    # Global outlet mask (all False initially)
    multi_outlet_mask = np.zeros(out_shape, dtype=bool)

    for geom in geometries:
        if geom.is_empty:
            continue

        # Rasterize the current sub-basin polygon into a temporary mask.
        temp_mask = rasterize(
            [(geom, 1)],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        ).astype(bool)

        # Get indices of pixels that are inside the current sub-basin.
        indices = np.where(temp_mask)
        if len(indices[0]) > 0:
            # Identify the DEM cell within this polygon that has the lowest elevation.
            min_index = np.argmin(DEM[temp_mask])
            outlet_i = indices[0][min_index]
            outlet_j = indices[1][min_index]
            # Mark that cell as the outlet
            multi_outlet_mask[outlet_i, outlet_j] = True
    return multi_outlet_mask