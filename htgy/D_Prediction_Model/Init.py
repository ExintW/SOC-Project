import os
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter
from globalss import *

# Append parent directory to path to access 'globals' if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from globals import *  # Expects DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

# ---------------------------------------------------------------------
# Helper: create 2D grid from CSV by pivoting lat/lon
# ---------------------------------------------------------------------
def create_grid(data, col_name):
    """
    Creates a 2D grid using pivot (lat descending, lon ascending).
    """
    return (
        data.pivot(index=MAP_STATS.lat_col, columns=MAP_STATS.lon_col, values=col_name)
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

def init_global_data_structs(fraction=1):
    # Read the Loess Plateau border shapefile and combine all features into one geometry.
    loess_border_path = DATA_DIR / "Loess_Plateau_vector_border.shp"
    loess_border = gpd.read_file(loess_border_path)

    print("Loess border reported CRS:", loess_border.crs)
    print("Loess border total_bounds:", loess_border.total_bounds)

    # union_all() merges all features; recommended over unary_union in newer geopandas versions.
    MAP_STATS.loess_border_geom = loess_border.union_all()

    # Reproject the Loess Plateau border to the desired CRS (if needed).
    loess_border = loess_border.to_crs(desired_crs)
    MAP_STATS.loess_border_geom = loess_border.union_all()


    # Define file paths for the region CSV, dam CSV, and SOC proportion CSV.
    region_csv_path = PROCESSED_DIR / "Resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
    dam_csv_path = PROCESSED_DIR / "htgy_Dam_with_matched_points.csv"
    proportion_csv_path = DATA_DIR / "Fast_Slow_SOC_Proportion.csv"

    # Read the CSV files.
    df = pd.read_csv(region_csv_path, encoding='utf-8-sig')
    MAP_STATS.df_dam = pd.read_csv(dam_csv_path, encoding='utf-8-sig')
    MAP_STATS.df_prop = pd.read_csv(proportion_csv_path, encoding='utf-8-sig')

    # Ensure numeric values for dam construction year and storage.
    MAP_STATS.df_dam["year"] = pd.to_numeric(MAP_STATS.df_dam["year"], errors="coerce")
    MAP_STATS.df_dam["total_stor"] = pd.to_numeric(MAP_STATS.df_dam["total_stor"], errors="coerce")
    MAP_STATS.df_dam["deposition"] = pd.to_numeric(MAP_STATS.df_dam["deposition"], errors="coerce")
    MAP_STATS.df_dam["capacity_remained"] = MAP_STATS.df_dam["total_stor"] - MAP_STATS.df_dam["deposition"]

    # ---------------------------------------------------------------------
    # Define column names from the region CSV.
    # ---------------------------------------------------------------------
    MAP_STATS.lon_col, MAP_STATS.lat_col = "LON", "LAT"
    soc_col = "ORGA"        # Initial SOC concentration (g/kg)
    dem_col = "htgy_DEM"    # DEM (elevation)
    landuse_col = "LANDUSE" # Land use
    region_col = "Region"   # Not used later
    slope_col = "SLOPE"     # Slope values
    k1_col = "SOC_k1_fast_pool (1/month)"  # Fast pool decay rate
    k2_col = "SOC_k2_slow_pool (1/month)"  # Slow pool decay rate

    # ---------------------------------------------------------------------
    # Extract grid coordinates
    # ---------------------------------------------------------------------
    MAP_STATS.grid_x = np.sort(df[MAP_STATS.lon_col].unique())           # ascending
    MAP_STATS.grid_y = np.sort(df[MAP_STATS.lat_col].unique())[::-1]     # descending for latitude

    # ---------------------------------------------------------------------
    # Create 2D arrays from CSV
    # ---------------------------------------------------------------------
    INIT_VALUES.SOC = create_grid(df, soc_col)  # SOC concentration (g/kg)
    INIT_VALUES.SOC_valid = INIT_VALUES.SOC.copy() # for validation
    INIT_VALUES.SOC *= fraction                 # for past simulation
    INIT_VALUES.SOC *= 10 * 0.58
    INIT_VALUES.SOC = np.clip(INIT_VALUES.SOC, None, C_INIT_CAP)    # Clip values above 12
    INIT_VALUES.DEM = create_grid(df, dem_col)
    INIT_VALUES.SAND = create_grid(df, "SAND")
    INIT_VALUES.SILT = create_grid(df, "SILT")
    INIT_VALUES.CLAY = create_grid(df, "CLAY")
    INIT_VALUES.LANDUSE = create_grid(df, landuse_col).astype(object)
    INIT_VALUES.REGION = create_grid(df, region_col).astype(object)
    INIT_VALUES.SLOPE = create_grid(df, slope_col)
    INIT_VALUES.K_fast = create_grid(df, k1_col)
    INIT_VALUES.K_slow = create_grid(df, k2_col)
    
    # =============================================================================
    # 2) PARTITION SOC INTO FAST & SLOW POOLS
    # =============================================================================
    INIT_VALUES.C_fast, INIT_VALUES.C_slow, MAP_STATS.p_fast_grid = allocate_fast_slow_soc()

    # =============================================================================
    # 3) Create a output matrix for total SOC
    # =============================================================================
    MAP_STATS.total_C_matrix = []
    
    print(f"Initial p_fast_grid mean = {np.nanmean(MAP_STATS.p_fast_grid)}, max = {np.nanmax(MAP_STATS.p_fast_grid)}, min = {np.nanmin(MAP_STATS.p_fast_grid)}")

import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter
from globals import PROCESSED_DIR

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

    # --- step 1: find 1 km low‐points ---
    fp       = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=bool)
    neigh_min = minimum_filter(dem, footprint=fp, mode="nearest")
    low_mask  = neigh_min > dem

    # --- step 2: load 10 m low‐point counts ---
    df_cnt     = pd.read_csv(PROCESSED_DIR / "Low_Point_Summary.csv", encoding="utf-8-sig")
    lon_to_i   = {lon: i for i, lon in enumerate(MAP_STATS.grid_x)}
    lat_to_j   = {lat: j for j, lat in enumerate(MAP_STATS.grid_y)}

    # --- step 3: build a full count matrix ---
    count_mat = np.zeros_like(dem, dtype=int)
    for _, row in df_cnt.iterrows():
        lon, lat, cnt = row["LON"], row["LAT"], int(row["low_10m_count"])
        i = lon_to_i.get(lon)
        j = lat_to_j.get(lat)
        if i is not None and j is not None:
            count_mat[j, i] = cnt

    # --- step 4: compute capacity using that count_mat ---
    height_diff = neigh_min - dem
    Low_Point_Capacity = np.zeros_like(dem, dtype=float)
    Low_Point_Capacity[low_mask] = (
        area
        * height_diff[low_mask]
        * count_mat[low_mask]
    )

    # --- step 5: dem difference matrix for reference ---
    Low_Point_DEM_Dif = np.zeros_like(dem, dtype=float)
    Low_Point_DEM_Dif[low_mask] = height_diff[low_mask]
    Low_Point_DEM_Dif[Low_Point_DEM_Dif == 0] = np.nan

    print(f"Low_Point_Capacity: max = {np.nanmax(Low_Point_Capacity):.2f}, min = {np.nanmin(Low_Point_Capacity):.2f}, and mean = {np.nanmean(Low_Point_Capacity):.2f}, and sum = {np.nansum(Low_Point_Capacity):.2f}")

    # Now Low_Point_Capacity[i,j] > 0 exactly at your low-lying points
    return low_mask, Low_Point_Capacity, Low_Point_DEM_Dif


def clean_nan():
    INIT_VALUES.SOC[~MAP_STATS.loess_border_mask] = np.nan
    INIT_VALUES.DEM[~MAP_STATS.loess_border_mask] = np.nan
    INIT_VALUES.SAND[~MAP_STATS.loess_border_mask] = np.nan
    INIT_VALUES.SILT[~MAP_STATS.loess_border_mask] = np.nan
    INIT_VALUES.CLAY[~MAP_STATS.loess_border_mask] = np.nan
    INIT_VALUES.LANDUSE[~MAP_STATS.loess_border_mask] = np.nan
    INIT_VALUES.REGION[~MAP_STATS.loess_border_mask] = np.nan
    INIT_VALUES.SLOPE[~MAP_STATS.loess_border_mask] = np.nan
    INIT_VALUES.K_fast[~MAP_STATS.loess_border_mask] = np.nan
    INIT_VALUES.K_slow[~MAP_STATS.loess_border_mask] = np.nan
    INIT_VALUES.C_fast[~MAP_STATS.loess_border_mask] = np.nan
    INIT_VALUES.C_slow[~MAP_STATS.loess_border_mask] = np.nan
    MAP_STATS.p_fast_grid[~MAP_STATS.loess_border_mask] = np.nan
    
    # Fill missing values in some arrays.
    INIT_VALUES.DEM[MAP_STATS.loess_border_mask] = np.nan_to_num(INIT_VALUES.DEM[MAP_STATS.loess_border_mask], nan=np.nanmean(INIT_VALUES.DEM))
    INIT_VALUES.SOC[MAP_STATS.loess_border_mask] = np.nan_to_num(INIT_VALUES.SOC[MAP_STATS.loess_border_mask], nan=np.nanmean(INIT_VALUES.SOC))
    INIT_VALUES.SAND[MAP_STATS.loess_border_mask] = np.nan_to_num(INIT_VALUES.SAND[MAP_STATS.loess_border_mask], nan=np.nanmean(INIT_VALUES.SAND))
    INIT_VALUES.SILT[MAP_STATS.loess_border_mask] = np.nan_to_num(INIT_VALUES.SILT[MAP_STATS.loess_border_mask], nan=np.nanmean(INIT_VALUES.SILT))
    INIT_VALUES.CLAY[MAP_STATS.loess_border_mask] = np.nan_to_num(INIT_VALUES.CLAY[MAP_STATS.loess_border_mask], nan=np.nanmean(INIT_VALUES.CLAY))
    #INIT_VALUES.LANDUSE[MAP_STATS.loess_border_mask] = np.nan_to_num(INIT_VALUES.LANDUSE[MAP_STATS.loess_border_mask], nan='not used')
    #INIT_VALUES.REGION[MAP_STATS.loess_border_mask] = np.nan_to_num(INIT_VALUES.REGION[MAP_STATS.loess_border_mask], nan='erosion area')
    INIT_VALUES.SLOPE[MAP_STATS.loess_border_mask] = np.nan_to_num(INIT_VALUES.SLOPE[MAP_STATS.loess_border_mask], nan=np.nanmean(INIT_VALUES.SLOPE))
    INIT_VALUES.K_fast[MAP_STATS.loess_border_mask] = np.nan_to_num(INIT_VALUES.K_fast[MAP_STATS.loess_border_mask], nan=np.nanmean(INIT_VALUES.K_fast))
    INIT_VALUES.K_slow[MAP_STATS.loess_border_mask] = np.nan_to_num(INIT_VALUES.K_slow[MAP_STATS.loess_border_mask], nan=np.nanmean(INIT_VALUES.K_slow))
    INIT_VALUES.C_fast[MAP_STATS.loess_border_mask] = np.nan_to_num(INIT_VALUES.C_fast[MAP_STATS.loess_border_mask], nan=np.nanmean(INIT_VALUES.C_fast))
    INIT_VALUES.C_slow[MAP_STATS.loess_border_mask] = np.nan_to_num(INIT_VALUES.C_slow[MAP_STATS.loess_border_mask], nan=np.nanmean(INIT_VALUES.C_slow))
    INIT_VALUES.LANDUSE[pd.isna(INIT_VALUES.LANDUSE)] = 'not used'
    INIT_VALUES.REGION[pd.isna(INIT_VALUES.REGION)] = 'erosion area'
