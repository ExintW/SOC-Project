import os
import sys
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from shapely.geometry import LineString, MultiLineString
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import Paths 
from global_structs import MAP_STATS, INIT_VALUES
from RUSLE_calculations import calculate_LS_factor, calculate_p_factor
from config import *

class TeeOutput:
    def __init__(self, file, original_stdout):
        self.file = file
        self.original_stdout = original_stdout
    
    def write(self, text):
        self.file.write(text)
        if ("Processing year" in text or 
            "=======================================================================" in text or
            "Year" in text and "Month" in text or
            "Completed simulation for Year" in text):
            self.original_stdout.write(text)
        self.file.flush()
    
    def flush(self):
        self.file.flush()
        self.original_stdout.flush()

def gaussian_blur_with_nan(data, sigma=1):
    mask = ~np.isnan(data)
    data_filled = np.where(mask, data, 0)

    blurred_data = ndimage.gaussian_filter(data_filled, sigma=sigma)
    blurred_mask = ndimage.gaussian_filter(mask.astype(float), sigma=sigma)

    with np.errstate(invalid='ignore'):
        result = blurred_data / blurred_mask
    result[~MAP_STATS.border_mask] = np.nan
    return result

def _nearest_index_1d(vals, coords):
    """
    Return indices of the nearest centres in `coords` for each value in `vals`,
    irrespective of whether `coords` is ascending or descending.
    """
    # Detect order
    asc = coords[0] < coords[-1]
    coords_sorted = coords if asc else coords[::-1]

    # Vectorised nearest‑neighbour search
    j = np.searchsorted(coords_sorted, vals)               # right neighbour
    j = np.clip(j, 1, coords_sorted.size - 1)              # keep 1‥N‑1
    left_closer = np.abs(vals - coords_sorted[j - 1]) <= np.abs(vals - coords_sorted[j])
    j = j - left_closer                                    # choose nearer (tie → lower index)

    # Map back to original order if axis was descending 
    return j if asc else (coords_sorted.size - 1 - j)

def create_grid_from_points(lon, lat, val, grid_x, grid_y):
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    val = np.asarray(val)

    # Allocate target grid (NaN‑filled)
    out = np.full((grid_y.size, grid_x.size), np.nan, dtype=val.dtype)

    # Find nearest column / row indices
    j = _nearest_index_1d(lon, grid_x)
    i = _nearest_index_1d(lat, grid_y)

    # Write values 
    out[i, j] = val
    return out

def resample_LS_to_grid(ls_DEM, factor=GRID_RESOLUTION // DEM_RESOLUTION):
    """
    Resample DEM resolution LS factor to grid resolution
    """
    h, w = ls_DEM.shape
    h_crop = h - h % factor
    w_crop = w - w % factor
    ls_DEM = ls_DEM[:h_crop, :w_crop]
    
    ls_grid = ls_DEM.reshape(h_crop // factor, factor, w_crop // factor, factor).mean(axis=(1, 3))
    
    h, w = ls_grid.shape
    th, tw = len(MAP_STATS.grid_y), len(MAP_STATS.grid_x)

    start_h = (h - th) // 2
    start_w = (w - tw) // 2

    return ls_grid[start_h:start_h + th, start_w:start_w + tw]

def compute_const_RUSLE():
    LS_factor = calculate_LS_factor(INIT_VALUES.SLOPE)
    LS_factor = resample_LS_to_grid(LS_factor)
    LS_factor = np.nan_to_num(LS_factor, nan=np.nanmean(LS_factor))
    LS_factor[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.LS_FACTOR = LS_factor
    
    P_factor = np.array([
        [calculate_p_factor(INIT_VALUES.LANDUSE[i, j], INIT_VALUES.SLOPE[i, j]) for j in range(INIT_VALUES.LANDUSE.shape[1])]
        for i in range(INIT_VALUES.LANDUSE.shape[0])
    ])
    P_factor = np.nan_to_num(P_factor, nan=np.nanmean(P_factor))
    P_factor[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.P_FACTOR = P_factor
    
def print_factor_info(factor, name="Factor"):
    """
    Debug print function for a factor and plot it on the map
    """
    total_elements = np.count_nonzero(~np.isnan(factor))
    max_value = np.nanmax(factor)
    min_value = np.nanmin(factor)
    mean_value = np.nanmean(factor)
    nan_count = np.count_nonzero(np.isnan(factor))
    print(f"Total elements in {name}: {total_elements}, with max = {max_value}, min = {min_value}, mean = {mean_value}, nan count = {nan_count}")
    
    # Plot figure
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        factor,
        cmap="viridis", vmin=0, vmax=max_value,
        extent=[
            MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(),
            MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()
        ],
        origin="upper"
    )

    # Overlay the border
    border = MAP_STATS.border_geom.boundary
    if isinstance(border, LineString):
        x, y = border.xy
        ax.plot(x, y, color="black", linewidth=0.4)
    elif isinstance(border, MultiLineString):
        for seg in border.geoms:
            x, y = seg.xy
            ax.plot(x, y, color="black", linewidth=0.4)

    # Append a colorbar axis the same height as the map, with padding
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad="4%")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f"{name} Value")

    # 4) Labels and formatting
    ax.set_title(f"{name} Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="x")

    # 5) Save and close
    filename_fig = f"{name}_plot.png"
    plt.savefig(
        os.path.join(Paths.OUTPUT_DIR, filename_fig),
        dpi=600,
        bbox_inches="tight"
    )
    plt.close(fig)