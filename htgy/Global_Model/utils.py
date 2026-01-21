import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds


from shapely.geometry import LineString, MultiLineString
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import r2_score

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

def create_grid_from_points(lon_src, lat_src, values, lon_grid, lat_grid):
    """
    Supports 2 cases:

    Case A) lon_src and lat_src are axis arrays (regular grid):
        lon_src = (nlon,), lat_src = (nlat,)
        values  = (nlat, nlon) OR flattened (nlat*nlon,)

    Case B) lon_src and lat_src are point arrays:
        lon_src = (N,), lat_src = (N,)
        values  = (N,)
    """

    lon_src = np.asarray(lon_src)
    lat_src = np.asarray(lat_src)
    values = np.asarray(values)

    out = np.full((len(lat_grid), len(lon_grid)), np.nan, dtype=float)

    # map source axis values to nearest indices in target grid
    j = np.array([np.argmin(np.abs(lon_grid - x)) for x in lon_src], dtype=int)
    i = np.array([np.argmin(np.abs(lat_grid - y)) for y in lat_src], dtype=int)

    # -----------------------------
    # Case A: lon_src + lat_src are axes (grid)
    # -----------------------------
    if lon_src.size != lat_src.size:
        # values can be 2D (nlat,nlon) or flattened (nlat*nlon,)
        if values.ndim == 1:
            values = values.reshape(lat_src.size, lon_src.size)

        out[np.ix_(i, j)] = values
        return out

    # -----------------------------
    # Case B: lon_src + lat_src are point lists
    # -----------------------------
    out[i, j] = values
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
    
def print_plot_factor_info(factor, name="Factor", max_val=None):
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
        cmap="viridis", vmin=0, vmax=max_val if max_val is not None else max_value,
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
    
def precompute_sorted_indices():
    """
    Precompute sorted indices of DEM cells in descending order of elevation.
    """
    rows, cols = INIT_VALUES.DEM.shape
    flat_dem = INIT_VALUES.DEM.flatten()
    sorted_flat_indices = np.argsort(flat_dem)[::-1]  # descending order
    sorted_indices = np.empty((sorted_flat_indices.shape[0], 2), dtype=np.int64)
    sorted_indices[:, 0], sorted_indices[:, 1] = np.unravel_index(sorted_flat_indices, (rows, cols))
    INIT_VALUES.SORTED_INDICES = sorted_indices
    
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

def export_dam_remained_cap_netcdf(dam_rem_cap_array, time_start, lat, lon, out_path):
    """
    Wrap a 3-D numpy array (time × lat × lon) into an xarray Dataset
    with a monthly time axis, and write to NetCDF.
    """
    n_steps = dam_rem_cap_array.shape[0]
    time_index = pd.date_range(start=f"{time_start}-01-01",
                               periods=n_steps,
                               freq="MS")
    da = xr.DataArray(
        dam_rem_cap_array,
        dims=("time", "lat", "lon"),
        coords={"time": time_index, "lat": lat, "lon": lon},
        name="dam_rem_cap"
    )
    ds = da.to_dataset()
    ds.to_netcdf(out_path)

def save_nc(start_year, end_year):
    # stack into an (X, 844, 1263) array
    total_C_array = np.stack(MAP_STATS.total_C_matrix, axis=0).astype(np.float32)
    dam_rem_cap_array = np.stack(MAP_STATS.dam_rem_cap_matrix, axis=0).astype(np.float32)

    C_nc_path = Paths.OUTPUT_DIR / f"Total_C_{start_year}-{end_year}_monthly.nc"
    export_total_C_netcdf(
        total_C_array,
        time_start=start_year,
        lat=MAP_STATS.grid_y,
        lon=MAP_STATS.grid_x,
        out_path=C_nc_path
    )
    dam_nc_path = Paths.OUTPUT_DIR / f"Dam_rem_Cap_{start_year}-{end_year}_monthly.nc"
    export_dam_remained_cap_netcdf(
        dam_rem_cap_array,
        time_start=start_year,
        lat=MAP_STATS.grid_y,
        lon=MAP_STATS.grid_x,
        out_path=dam_nc_path
    )

def validate_SOC(pred, true):
    mask = ~np.isnan(true)

    # Flatten both arrays using the mask
    y_true = true[mask]
    y_pred = pred[mask]

    # Compute R²
    r2 = r2_score(y_true, y_pred)
    print(f"Past SOC Validation: R^2 score: {r2}")

def resolve_cmip6_lai_segment(year, segments):
    """
    Return (lai_file_path, cmip_start) for `year` using segments like:
      {"start_year": ..., "end_year": int|None, "cmip_start": ..., "relpath": Path}
    Assumes segments are sorted by start_year and non-overlapping.
    """
    for s in segments:
        if year < s["start_year"]:
            break
        end = s["end_year"]
        if end is None or year <= end:
            return s["relpath"], s["cmip_start"]
    raise ValueError(f"No CMIP6 LAI segment configured for year {year}.")

def find_nearest_index(array, value):
    """Return index of element in array closest to value."""
    return (np.abs(array - value)).argmin()

def init_dams(year):
    active_dams = np.zeros(INIT_VALUES.DEM.shape, dtype=int)
    full_dams = np.zeros(INIT_VALUES.DEM.shape, dtype=int)
    dam_max_cap = np.zeros(INIT_VALUES.DEM.shape, dtype=np.float64)
    dam_cur_stored = MAP_STATS.dam_cur_stored
    
    MAP_STATS.active_dams = active_dams
    MAP_STATS.full_dams = full_dams
    MAP_STATS.dam_max_cap = dam_max_cap
    MAP_STATS.dam_cur_stored = dam_cur_stored
    
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

def plot_SOC(soc, year, month_idx, ext=None):
    fig, ax = plt.subplots()
    im = ax.imshow(soc, cmap="viridis", vmin=0,vmax=30,
                    extent=[MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(), MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()],
                    origin='upper')
    # overlay the border (no fill, just outline)
    border = MAP_STATS.border_geom.boundary

    if isinstance(border, LineString):
        x, y = border.xy
        ax.plot(x, y, color="black", linewidth=0.4)
    elif isinstance(border, MultiLineString):
        for seg in border.geoms:
            x, y = seg.xy
            ax.plot(x, y, color="black", linewidth=0.4)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad="4%")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("SOC (g/kg)")
    if ext is None:
        ax.set_title(f"SOC at Timestep Year {year}, Month {month_idx+1}")
    else:
        ax.set_title(f"SOC at Timestep Year {year}, Month {month_idx+1}, {ext}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='x')
    if ext is None:
        filename_fig = f"SOC_{year}_{month_idx+1:02d}_River.png"
    else:
        filename_fig = f"SOC_{year}_{month_idx+1:02d}_{ext}_River.png"
    plt.savefig(os.path.join(Paths.OUTPUT_DIR, "Figure", filename_fig), dpi=600)
    plt.close("all")

def plot_SOC_timestep(year, month_idx):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot SOC
    im = ax.imshow(
        MAP_STATS.C_fast_current + MAP_STATS.C_slow_current,
        cmap="viridis", vmin=0, vmax=250,
        extent=[
            MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(),
            MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()
        ],
        origin="upper"
    )

    # Overlay the border (no fill, just outline)
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
    cbar.set_label("SOC (g/kg)")

    # Labels and formatting
    ax.set_title(f"SOC at Timestep Year {year}, Month {month_idx + 1}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="x")

    # Save and close
    filename_fig = f"SOC_{year}_{month_idx + 1:02d}_River.png"
    plt.savefig(
        os.path.join(Paths.OUTPUT_DIR, "Figure", filename_fig),
        dpi=600,
        bbox_inches="tight"
    )
    plt.close(fig)

def resample_tif_to_model_grid(tif_path, grid_x, grid_y, resampling=Resampling.bilinear):
    """
    Read a GeoTIFF and resample it to the model grid defined by grid_x and grid_y.

    Returns:
        out_2d: shape (len(grid_y), len(grid_x))
    """

    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)

    nx = len(grid_x)
    ny = len(grid_y)

    # grid resolution (assumes regular spacing)
    dx = float(np.median(np.diff(np.sort(grid_x))))
    dy = float(np.median(np.diff(np.sort(grid_y))))

    left   = float(grid_x.min() - dx / 2.0)
    right  = float(grid_x.max() + dx / 2.0)
    bottom = float(grid_y.min() - dy / 2.0)
    top    = float(grid_y.max() + dy / 2.0)

    # rasterio transform expects row 0 at "top" (largest y)
    dst_transform = from_bounds(left, bottom, right, top, nx, ny)

    out_2d = np.full((ny, nx), np.nan, dtype=np.float32)

    with rasterio.open(tif_path) as src:
        src_data = src.read(1).astype(np.float32)

        # convert nodata to NaN
        if src.nodata is not None:
            src_data = np.where(src_data == src.nodata, np.nan, src_data)

        reproject(
            source=src_data,
            destination=out_2d,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",   # assumes your grid_x/grid_y are lon/lat
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )

    # If your MAP_STATS.grid_y is ascending (south to north),
    # flip because rasterio rows start from top (north).
    if grid_y[0] < grid_y[-1]:
        out_2d = np.flipud(out_2d)

    return out_2d

def fill_nan_with_mean(arr, mask=None):
    """
    Fill NaN values with mean.
    If mask is provided, mean is computed only within mask.
    """
    out = arr.copy()
    if mask is None:
        mean_val = np.nanmean(out)
        out = np.nan_to_num(out, nan=mean_val)
    else:
        mean_val = np.nanmean(out[mask])
        out[mask] = np.nan_to_num(out[mask], nan=mean_val)
    return out