import numpy as np
from globalss import *

def find_nearest_index(array, value):
    """Return index of element in array closest to value."""
    return (np.abs(array - value)).argmin()

# =============================================================================
# CONVERT SOIL LOSS TO SOC LOSS (g/kg/month)
# =============================================================================
def convert_soil_loss_to_soc_loss_monthly(E_t_ha_month, ORGA_g_per_kg, bulk_density=1300):
    """
    Convert soil loss (t/ha/month) to SOC loss (g/kg/month).
    1 t/ha = 100 g/m². Then multiply by (SOC_concentration / 1000) * bulk_density.
    """
    E_g_m2_month = E_t_ha_month * 100.0
    soc_loss_g_m2_month = E_g_m2_month * (ORGA_g_per_kg / 1000.0) * bulk_density
    return soc_loss_g_m2_month / bulk_density

# =============================================================================
# HELPER: REGRID CMIP/ERA5 POINT DATA TO 2D GRID
# =============================================================================
def create_grid_from_points(lon_points, lat_points, values, grid_x, grid_y):
    """
    Regrid 1D point data to a 2D grid by assigning each point to the nearest cell center.
    """
    grid = np.full((len(grid_y), len(grid_x)), np.nan)
    for k in range(len(values)):
        j = (np.abs(grid_x - lon_points[k])).argmin()
        i = (np.abs(grid_y - lat_points[k])).argmin()
        grid[i, j] = values[k]
    return grid

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