import os
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
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
            'fast': row['Fast SOC(%)'] / 100,
            'slow': row['Slow SOC(%)'] / 100
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
    region_csv_path = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
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
    INIT_VALUES.SOC = np.clip(INIT_VALUES.SOC, None, 12)    # Clip values above 12
    INIT_VALUES.DEM = create_grid(df, dem_col)
    INIT_VALUES.SAND = create_grid(df, "SAND")
    INIT_VALUES.SILT = create_grid(df, "SILT")
    INIT_VALUES.CLAY = create_grid(df, "CLAY")
    INIT_VALUES.LANDUSE = create_grid(df, landuse_col)
    INIT_VALUES.REGION = create_grid(df, region_col)
    INIT_VALUES.SLOPE = create_grid(df, slope_col)
    INIT_VALUES.K_fast = create_grid(df, k1_col)
    INIT_VALUES.K_slow = create_grid(df, k2_col)

    # Fill missing values in some arrays.
    INIT_VALUES.DEM = np.nan_to_num(INIT_VALUES.DEM, nan=np.nanmean(INIT_VALUES.DEM))
    INIT_VALUES.SLOPE = np.nan_to_num(INIT_VALUES.SLOPE, nan=np.nanmean(INIT_VALUES.SLOPE))
    INIT_VALUES.K_fast = np.nan_to_num(INIT_VALUES.K_fast, nan=np.nanmean(INIT_VALUES.K_fast))
    INIT_VALUES.K_slow = np.nan_to_num(INIT_VALUES.K_slow, nan=np.nanmean(INIT_VALUES.K_slow))
    
    # =============================================================================
    # 2) PARTITION SOC INTO FAST & SLOW POOLS
    # =============================================================================
    INIT_VALUES.C_fast, INIT_VALUES.C_slow, MAP_STATS.p_fast_grid = allocate_fast_slow_soc()