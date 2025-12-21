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

def precompute_river_basin(cache_path=None):
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