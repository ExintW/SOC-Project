import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import netCDF4 as nc
from scipy.ndimage import minimum_filter

from config import *
from global_structs import INIT_VALUES, MAP_STATS
from utils import create_grid_from_points

def create_grid(data, col_name):
    """
    Creates a 2D grid using pivot (lat descending, lon ascending).
    """
    return (
        data.pivot(index=LAT_COL, columns=LON_COL, values=col_name)
        .sort_index(ascending=False)
        .values
    )
    
def allocate_fast_slow_soc():
    """
    Partition total SOC into fast and slow pools using percentages from the CSV
    (e.g., farmland might have 30% fast pool, 70% slow).
    """
    prop_dict = {
        row['Type']: {
            'fast': row['Fast SOC(%)'] / 100 / P_FAST_DIV_FACTOR,
            'slow': 1 - (row['Fast SOC(%)'] / 100 / P_FAST_DIV_FACTOR)
        }
        for _, row in MAP_STATS.df_prop.iterrows()
    }
    rows, cols = INIT_VALUES.LANDUSE.shape
    C_fast = np.zeros((rows, cols))
    C_slow = np.zeros((rows, cols))
    p_fast_grid = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            land_type = INIT_VALUES.LANDUSE[i, j]
            props = prop_dict.get(land_type, {'fast': 0, 'slow': 1})
            C_fast[i, j] = INIT_VALUES.SOC[i, j] * props['fast']
            C_slow[i, j] = INIT_VALUES.SOC[i, j] * props['slow']
            p_fast_grid[i, j] = props['fast']
    return C_fast, C_slow, p_fast_grid

def init_global_data_structs():
    # =============================================================================
    # LOAD REGION BORDER 
    # =============================================================================
    # Read in the Loess Plateau border shapefile
    region_border = gpd.read_file(BORDER_SHP)
    
    # Reproject the border to the desired CRS if necessary
    if region_border.crs != DESIRED_CRS:
        region_border = region_border.to_crs(DESIRED_CRS)
    
    # Merge all features
    MAP_STATS.border_geom = region_border.union_all()
    
    # =============================================================================
    # LOAD REGION FEATURES 
    # =============================================================================
    df = pd.read_csv(INIT_SOC_CSV)
    df_dam = pd.read_csv(DAM_CSV)
    MAP_STATS.df_prop = pd.read_csv(FAST_SLOW_RATIO_CSV)
    
    # Ensure numeric values for dam construction year and storage
    df_dam[DAM_YEAR] = pd.to_numeric(df_dam[DAM_YEAR], errors="coerce")
    df_dam[DAM_STORAGE] = pd.to_numeric(df_dam[DAM_STORAGE], errors="coerce")
    df_dam[DAM_DEPOSITION] = pd.to_numeric(df_dam[DAM_DEPOSITION], errors="coerce")
    df_dam[DAM_CAPACITY_REMAINED] = df_dam[DAM_STORAGE] - df_dam[DAM_DEPOSITION]
    MAP_STATS.df_dam = df_dam

    # =============================================================================
    # PROCESS GRID 
    # =============================================================================
    MAP_STATS.grid_x = np.sort(df[LON_COL].unique())           # ascending
    MAP_STATS.grid_y = np.sort(df[LAT_COL].unique())[::-1]     # descending for latitude
    
    # Create 2D arrays from CSV
    SOC = create_grid(df, SOC_COL)  # SOC concentration (g/kg)
    SOC *= C_INIT_FACTOR
    if C_INIT_CAP > 0:
        SOC = np.clip(SOC, None, C_INIT_CAP) 
    INIT_VALUES.SOC = SOC
    INIT_VALUES.DEM = create_grid(df, DEM_COL)
    INIT_VALUES.SAND = create_grid(df, SAND_COL)
    INIT_VALUES.SILT = create_grid(df, SILT_COL)
    INIT_VALUES.CLAY = create_grid(df, CLAY_COL)
    INIT_VALUES.LANDUSE = create_grid(df, LANDUSE_COL).astype(object)
    INIT_VALUES.REGION = create_grid(df, REGION_COL).astype(object)
    INIT_VALUES.SLOPE = create_grid(df, SLOPE_COL)
    INIT_VALUES.K_fast = create_grid(df, K1_COL)
    INIT_VALUES.K_slow = create_grid(df, K2_COL)
    
    # Partition SOC into fast and slow pools based on region proportions
    INIT_VALUES.C_fast, INIT_VALUES.C_slow, MAP_STATS.p_fast_grid = allocate_fast_slow_soc()
    
    # Load past SOC data
    with np.load(PAST_SOC_NPZ) as data:
        soc_past_fast = data['soc_mean_matrix']
        soc_past_fast = np.flipud(soc_past_fast)
        soc_past_fast = np.nan_to_num(soc_past_fast, nan=np.nanmean(soc_past_fast))
        soc_past_fast *= MAP_STATS.p_fast_grid
        INIT_VALUES.SOC_PAST_FAST = soc_past_fast * SOC_PAST_FACTOR
        soc_past_slow = data['soc_mean_matrix']
        soc_past_slow = np.flipud(soc_past_slow)
        soc_past_slow = np.nan_to_num(soc_past_slow, nan=np.nanmean(soc_past_slow))
        soc_past_slow *= (1 - MAP_STATS.p_fast_grid)
        INIT_VALUES.SOC_PAST_SLOW = soc_past_slow * SOC_PAST_FACTOR
        
    # Initialize current dam capacity
    MAP_STATS.dam_cur_stored = np.zeros(INIT_VALUES.DEM.shape, dtype=np.float64)
    
    # Precompute low-point masks and capacities
    MAP_STATS.low_mask, MAP_STATS.Low_Point_Capacity, MAP_STATS.Low_Point_DEM_Dif = precompute_low_point()

def precompute_low_point():
    """
    1) Identify 1 km low‐points in INIT_VALUES.DEM
    2) Load the per‐cell 10 m low‐point counts from CSV
    3) Build a full count_mat matching DEM shape
    4) Compute capacity = area * height_diff * count_mat
    5) Compute DEM difference matrix
    Returns: low_mask, Low_Point_Capacity, Low_Point_DEM_Dif
    """

    dem = INIT_VALUES.DEM  # shape (nrows, ncols)

    # 1 km cell area (m²)
    area = 10 * 10

    # find 1 km low‐points
    fp       = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=bool)
    neigh_min = minimum_filter(dem, footprint=fp, mode="nearest")
    low_mask  = neigh_min > dem

    # load 10 m low‐point counts
    df_cnt     = pd.read_csv(Paths.PROCESSED_DIR / "Low_Point_Summary.csv", encoding="utf-8-sig")
    lon_to_i   = {lon: i for i, lon in enumerate(MAP_STATS.grid_x)}
    lat_to_j   = {lat: j for j, lat in enumerate(MAP_STATS.grid_y)}

    # build a full count matrix
    count_mat = np.zeros_like(dem, dtype=int)
    for _, row in df_cnt.iterrows():
        lon, lat, cnt = row["LON"], row["LAT"], int(row["low_10m_count"])
        i = lon_to_i.get(lon)
        j = lat_to_j.get(lat)
        if i is not None and j is not None:
            count_mat[j, i] = cnt

    # compute capacity using that count_mat
    height_diff = neigh_min - dem
    Low_Point_Capacity = np.zeros_like(dem, dtype=float)
    Low_Point_Capacity[low_mask] = (
        area
        * height_diff[low_mask]
        * count_mat[low_mask]
    )

    # dem difference matrix for reference
    Low_Point_DEM_Dif = np.zeros_like(dem, dtype=float)
    Low_Point_DEM_Dif[low_mask] = height_diff[low_mask]
    Low_Point_DEM_Dif[Low_Point_DEM_Dif == 0] = np.nan

    print(f"Low_Point_Capacity: max = {np.nanmax(Low_Point_Capacity):.2f}, min = {np.nanmin(Low_Point_Capacity):.2f}, mean = {np.nanmean(Low_Point_Capacity):.2f}, and sum = {np.nansum(Low_Point_Capacity):.2f}")

    # Now Low_Point_Capacity[i,j] > 0 exactly at your low-lying points
    return low_mask, Low_Point_Capacity, Low_Point_DEM_Dif

def get_PAST_LAI():
    """
    Load past LAI trend and use it as one factor of prior for regularization.
    """
    lai_file = LAI_PAST_FILE
    cmip_start = CMIP_START
    with nc.Dataset(lai_file) as ds_lai:
        lon_lai = ds_lai.variables[LAI_LON][:] 
        lat_lai = ds_lai.variables[LAI_LAT][:]
        lai_data = ds_lai.variables[LAI_VAR][:]
        for month in range(12):
            idx = (PAST_KNOWN - cmip_start) * 12 + month
            lai_1d = lai_data[idx, :]
            LAI_2D = create_grid_from_points(lon_lai, lat_lai, lai_1d, MAP_STATS.grid_x, MAP_STATS.grid_y)
            LAI_2D = np.nan_to_num(LAI_2D, nan=np.nanmean(LAI_2D))
            LAI_2D[~MAP_STATS.border_mask] = np.nan
            INIT_VALUES.LAI_PAST.append(np.nanmean(LAI_2D))

def clean_nan():
    INIT_VALUES.SOC[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.DEM[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.SAND[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.SILT[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.CLAY[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.LANDUSE[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.REGION[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.SLOPE[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.K_fast[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.K_slow[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.C_fast[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.SOC_PAST_FAST[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.C_slow[~MAP_STATS.border_mask] = np.nan
    INIT_VALUES.SOC_PAST_SLOW[~MAP_STATS.border_mask] = np.nan
    MAP_STATS.p_fast_grid[~MAP_STATS.border_mask] = np.nan
    
    # Fill missing values in some arrays.
    INIT_VALUES.DEM[MAP_STATS.border_mask] = np.nan_to_num(INIT_VALUES.DEM[MAP_STATS.border_mask], nan=np.nanmean(INIT_VALUES.DEM))
    INIT_VALUES.SOC[MAP_STATS.border_mask] = np.nan_to_num(INIT_VALUES.SOC[MAP_STATS.border_mask], nan=np.nanmean(INIT_VALUES.SOC))
    INIT_VALUES.SAND[MAP_STATS.border_mask] = np.nan_to_num(INIT_VALUES.SAND[MAP_STATS.border_mask], nan=np.nanmean(INIT_VALUES.SAND))
    INIT_VALUES.SILT[MAP_STATS.border_mask] = np.nan_to_num(INIT_VALUES.SILT[MAP_STATS.border_mask], nan=np.nanmean(INIT_VALUES.SILT))
    INIT_VALUES.CLAY[MAP_STATS.border_mask] = np.nan_to_num(INIT_VALUES.CLAY[MAP_STATS.border_mask], nan=np.nanmean(INIT_VALUES.CLAY))
    INIT_VALUES.SLOPE[MAP_STATS.border_mask] = np.nan_to_num(INIT_VALUES.SLOPE[MAP_STATS.border_mask], nan=np.nanmean(INIT_VALUES.SLOPE))
    INIT_VALUES.K_fast[MAP_STATS.border_mask] = np.nan_to_num(INIT_VALUES.K_fast[MAP_STATS.border_mask], nan=np.nanmean(INIT_VALUES.K_fast))
    INIT_VALUES.K_slow[MAP_STATS.border_mask] = np.nan_to_num(INIT_VALUES.K_slow[MAP_STATS.border_mask], nan=np.nanmean(INIT_VALUES.K_slow))
    INIT_VALUES.C_fast[MAP_STATS.border_mask] = np.nan_to_num(INIT_VALUES.C_fast[MAP_STATS.border_mask], nan=np.nanmean(INIT_VALUES.C_fast))
    INIT_VALUES.C_slow[MAP_STATS.border_mask] = np.nan_to_num(INIT_VALUES.C_slow[MAP_STATS.border_mask], nan=np.nanmean(INIT_VALUES.C_slow))
    INIT_VALUES.LANDUSE[pd.isna(INIT_VALUES.LANDUSE)] = 'not used'
    INIT_VALUES.REGION[pd.isna(INIT_VALUES.REGION)] = 'erosion area'
