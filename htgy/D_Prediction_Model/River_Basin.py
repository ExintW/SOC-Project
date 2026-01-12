import os
import sys
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from affine import Affine
from rasterio.features import rasterize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import Paths
from config import *
from global_structs import INIT_VALUES, MAP_STATS

def precompute_river_basin():
    """
    Precompute (or load) river-basin masks and Loess Plateau border mask with unique IDs.
    Saves to / loads from a compressed .npz to skip heavy GIS work on subsequent runs.
    """
    
    # =============================================================================
    # TRY LOADING FROM CACHE
    # =============================================================================
    cache_path = Paths.PROCESSED_DIR / "precomputed_masks.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        MAP_STATS.small_boundary_mask = data["small_boundary_mask"]
        MAP_STATS.large_boundary_mask = data["large_boundary_mask"]
        MAP_STATS.river_mask = data["river_mask"]
        MAP_STATS.border_mask = data["border_mask"]
        MAP_STATS.small_outlet_mask = data["small_outlet_mask"]
        MAP_STATS.large_outlet_mask = data["large_outlet_mask"]
        print(f"Loaded precomputed masks from {cache_path}")
        return
    
    # =============================================================================
    # COMPUT RIVER BASIN MASKS
    # =============================================================================
    dx = np.mean(np.diff(MAP_STATS.grid_x))
    dy = abs(np.mean(np.diff(MAP_STATS.grid_y)))
    transform = Affine.translation(MAP_STATS.grid_x[0], MAP_STATS.grid_y[0]) * Affine.scale(dx, -dy)
    out_shape = (len(MAP_STATS.grid_y), len(MAP_STATS.grid_x))
    border = MAP_STATS.border_geom
    
    # Load border files
    border_gdf = gpd.GeoDataFrame(geometry=[border], crs=DESIRED_CRS)
    border_shp = gpd.read_file(BORDER_SHP).to_crs(DESIRED_CRS)
    border_clip = gpd.clip(border_shp, border_gdf).intersection(border)
    border_clip = border_clip.explode(index_parts=True).reset_index(drop=True)
    MAP_STATS.border_mask = rasterize(
        [(geom, 1) for geom in border_clip.geometry],
        out_shape=out_shape, transform=transform,
        fill=0, dtype=np.uint8, all_touched=True
    ).astype(bool)
    
    # Load River and River Basin files
    print("Masks computed:")
    print(f"  border_mask: {np.count_nonzero(MAP_STATS.border_mask)} pixels (True)")
    if SMALL_RIVER_BASIN_SHP is not None:
        small_boundary_shp = gpd.read_file(SMALL_RIVER_BASIN_SHP).to_crs(DESIRED_CRS)
        small_clip = gpd.clip(small_boundary_shp, border_gdf).intersection(border)
        small_clip = small_clip.explode(index_parts=True).reset_index(drop=True)
        small_shapes = [(geom, idx+1) for idx, geom in enumerate(small_clip.geometry)]
        MAP_STATS.small_boundary_mask = rasterize(
            small_shapes, out_shape=out_shape, transform=transform,
            fill=0, dtype=np.int32, all_touched=True
        )
        MAP_STATS.small_outlet_mask = compute_multi_outlet_mask(
            geometries=small_clip.geometry,
            DEM=INIT_VALUES.DEM,
            out_shape=out_shape,
            transform=transform
        )
        print(f"  small_boundary_mask: {np.count_nonzero(MAP_STATS.small_boundary_mask)} pixels (ID>0)")
        print(f"  small_outlet_mask: {np.count_nonzero(MAP_STATS.small_outlet_mask)} pixels (True)")
    if LARGE_RIVER_BASIN_SHP is not None:
        large_boundary_shp = gpd.read_file(LARGE_RIVER_BASIN_SHP).to_crs(DESIRED_CRS)
        large_clip = gpd.clip(large_boundary_shp, border_gdf).intersection(border)
        large_clip = large_clip.explode(index_parts=True).reset_index(drop=True)
        large_shapes = [(geom, idx+1) for idx, geom in enumerate(large_clip.geometry)]
        MAP_STATS.large_boundary_mask = rasterize(
            large_shapes, out_shape=out_shape, transform=transform,
            fill=0, dtype=np.int32, all_touched=True
        )
        MAP_STATS.large_outlet_mask = compute_multi_outlet_mask(
            geometries=large_clip.geometry,
            DEM=INIT_VALUES.DEM,
            out_shape=out_shape,
            transform=transform
        )
        print(f"  large_boundary_mask: {np.count_nonzero(MAP_STATS.large_boundary_mask)} pixels (ID>0)")
        print(f"  large_outlet_mask: {np.count_nonzero(MAP_STATS.large_outlet_mask)} pixels (True)")
    if RIVERS_SHP is not None:
        river_shp = gpd.read_file(RIVERS_SHP).to_crs(DESIRED_CRS)
        river_clip = gpd.clip(river_shp, border_gdf).intersection(border)
        river_clip = river_clip.explode(index_parts=True).reset_index(drop=True)
        MAP_STATS.river_mask = rasterize(
            [(geom, 1) for geom in river_clip.geometry],
            out_shape=out_shape, transform=transform,
            fill=0, dtype=np.uint8, all_touched=True
        ).astype(bool)
        print(f"  river_mask: {np.count_nonzero(MAP_STATS.river_mask)} pixels (True)")
    
    # Save all masks to cache
    np.savez_compressed(
        cache_path,
        small_boundary_mask=MAP_STATS.small_boundary_mask,
        large_boundary_mask=MAP_STATS.large_boundary_mask,
        river_mask=MAP_STATS.river_mask,
        border_mask=MAP_STATS.border_mask,
        small_outlet_mask=MAP_STATS.small_outlet_mask,
        large_outlet_mask=MAP_STATS.large_outlet_mask,
    )
    print(f"Computed and saved masks to {cache_path}")
    
# Helper function to compute multi-outlet mask
def compute_multi_outlet_mask(geometries, DEM, out_shape, transform):
    """
    Compute an outlet for each individual polygon provided in the 'geometries' iterable.
    For each polygon (sub-basin), rasterize it, then within that temporary mask find the DEM cell with the
    lowest value. Mark that cell in the global outlet mask.
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