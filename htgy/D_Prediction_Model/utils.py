import os
import sys
import numpy as np
import numba as nb
from globalss import *
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString
import matplotlib.ticker as mticker
import scipy.ndimage as ndimage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # Expects DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

def find_nearest_index(array, value):
    """Return index of element in array closest to value."""
    return (np.abs(array - value)).argmin()

def gaussian_blur_with_nan(data, sigma=1):
    mask = ~np.isnan(data)
    data_filled = np.where(mask, data, 0)

    blurred_data = ndimage.gaussian_filter(data_filled, sigma=sigma)
    blurred_mask = ndimage.gaussian_filter(mask.astype(float), sigma=sigma)

    with np.errstate(invalid='ignore'):
        result = blurred_data / blurred_mask
    result[~MAP_STATS.loess_border_mask] = np.nan
    return result

def plot_SOC(soc, year, month_idx):
    fig, ax = plt.subplots()
    cax = ax.imshow(soc, cmap="viridis", vmin=0,vmax=30,
                    extent=[MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(), MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()],
                    origin='upper')
    # overlay the border (no fill, just outline)
    border = MAP_STATS.loess_border_geom.boundary

    if isinstance(border, LineString):
        x, y = border.xy
        ax.plot(x, y, color="black", linewidth=0.4)
    elif isinstance(border, MultiLineString):
        for seg in border.geoms:
            x, y = seg.xy
            ax.plot(x, y, color="black", linewidth=0.4)
    cbar = fig.colorbar(cax, label="SOC (g/kg)")
    ax.set_title(f"SOC at Timestep Year {year}, Month {month_idx+1}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='x')
    filename_fig = f"SOC_{year}_{month_idx+1:02d}_River.png"
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure", filename_fig), dpi=600)
    plt.close("all")

# =============================================================================
# CONVERT SOIL LOSS TO SOC LOSS (g/kg/month)
# =============================================================================
def convert_soil_loss_to_soc_loss_monthly(E_t_ha_month, ORGA_g_per_kg, bulk_density=1300):
    """
    Convert soil loss (t/ha/month) to SOC loss (g/kg/month).
    1 t/ha = 100 g/m². Then multiply by (SOC_concentration / 1000) * bulk_density.
    """
    depth = 0.2
    E_g_m2_month = E_t_ha_month * 100.0
    soc_loss_g_per_kg = (E_g_m2_month / 1000.0) * ORGA_g_per_kg/depth / bulk_density
    return soc_loss_g_per_kg

def convert_soil_to_soc_loss(E_t_ha_month):
    return (E_t_ha_month * 0.1) / (DEPTH * BULK_DENSITY)

# =============================================================================
# HELPER: REGRID CMIP/ERA5 POINT DATA TO 2D GRID
# =============================================================================
def _nearest_index_1d(vals, coords):
    """
    Return indices of the nearest centres in `coords` for each value in `vals`,
    irrespective of whether `coords` is ascending or descending.
    """
    # Detect order ------------------------------------------------------------
    asc = coords[0] < coords[-1]
    coords_sorted = coords if asc else coords[::-1]

    # Vectorised nearest‑neighbour search -------------------------------------
    j = np.searchsorted(coords_sorted, vals)               # right neighbour
    j = np.clip(j, 1, coords_sorted.size - 1)              # keep 1‥N‑1
    left_closer = np.abs(vals - coords_sorted[j - 1]) <= np.abs(vals - coords_sorted[j])
    j = j - left_closer                                    # choose nearer (tie → lower index)

    # Map back to original order if axis was descending -----------------------
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

    # Write values (later duplicates overwrite earlier ones, like the loop)
    out[i, j] = val
    return out

# def create_grid_from_points(lon_points, lat_points, values, grid_x, grid_y):
#     """
#     Regrid 1D point data to a 2D grid by assigning each point to the nearest cell center.
#     """
#     grid = np.full((len(grid_y), len(grid_x)), np.nan)
#     for k in range(len(values)):
#         j = (np.abs(grid_x - lon_points[k])).argmin()
#         i = (np.abs(grid_y - lat_points[k])).argmin()
#         grid[i, j] = values[k]
#     return grid

def resample_LS_to_1km_grid(ls_30m, factor=33):
    """
    Resample 30m resolution LS factor to 1km grid
    """
    h, w = ls_30m.shape
    h_crop = h - h % factor
    w_crop = w - w % factor
    ls_30m = ls_30m[:h_crop, :w_crop]
    
    ls_1km = ls_30m.reshape(h_crop // factor, factor, w_crop // factor, factor).mean(axis=(1, 3))
    
    h, w = ls_1km.shape
    th, tw = len(MAP_STATS.grid_y), len(MAP_STATS.grid_x)

    start_h = (h - th) // 2
    start_w = (w - tw) // 2

    return ls_1km[start_h:start_h + th, start_w:start_w + tw]

def export_total_C_netcdf(total_C_array, time_start, lat, lon, out_path):
    """
    Wrap a 3-D numpy array (time × lat × lon) into an xarray Dataset
    with a monthly time axis, and write to NetCDF.
    """
    n_steps = total_C_array.shape[0]
    time_index = pd.date_range(start=f"{time_start}-01-01",
                               periods=n_steps,
                               freq="MS")
    da = xr.DataArray(
        total_C_array,
        dims=("time", "lat", "lon"),
        coords={"time": time_index, "lat": lat, "lon": lon},
        name="total_C"
    )
    ds = da.to_dataset()
    ds.to_netcdf(out_path)
    print(f"– Wrote NetCDF to {out_path}")