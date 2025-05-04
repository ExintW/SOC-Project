import numpy as np

USE_PARQUET = True  # Save output df as parquet instead of csv
CLEAN_OUTDIR = True

desired_crs = "EPSG:4326"
#desired_crs = "EPSG:3857"
BULK_DENSITY = 1300  # Convert dam capacity from 10,000 m³ to tons using a bulk density of 1300 t/m³.
DEPTH = 0.2
CELL_AREA_HA = 100.0  # 1 km² = 100 ha
M_soil = 2.6e8  # total soil mass per cell (kg)
INIT_YEAR = 2007
PRESENT_YEAR = 2025
# global_timestep = 0

class INIT_VALUES:
    SOC = None
    SOC_valid = None
    DEM = None
    SAND = None
    SILT = None
    CLAY = None
    LANDUSE = None
    REGION = None
    SLOPE = None
    K_fast = None
    K_slow = None
    C_fast = None
    C_slow = None
    
    @classmethod
    def reset(cls):
        for key in list(cls.__dict__):
            if not key.startswith('__') and not callable(getattr(cls, key)):
                setattr(cls, key, None)
    

class MAP_STATS:
    df_dam = None
    df_prop = None
    
    grid_x = None
    grid_y = None

    loess_border_geom = None
    
    lat_col = None
    lon_col = None
    
    p_fast_grid = None
    
    large_outlet_mask = None
    small_outlet_mask = None
    
    small_boundary_mask = None
    large_boundary_mask = None

    loess_border_mask = None
    river_mask = None
    
    C_fast_current = None
    C_slow_current = None
    
    @classmethod
    def reset(cls):
        for key in list(cls.__dict__):
            if not key.startswith('__') and not callable(getattr(cls, key)):
                setattr(cls, key, None)