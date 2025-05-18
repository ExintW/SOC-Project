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

############################Parameters##############################
C_INIT_CAP = 12

LAMBDA_FAST = 0       # for damping, set to 0 to disable
LAMBDA_SLOW = 0       # for damping, set to 0 to disable

ALPHA = 0.20          # for humification -> % minerized C fast that becomes C slow  (0 to disable)

A_MAX = 0.01

V_FAST_PROP = 0.8     # for vegetation input proportion
V_FACTOR = 3          # for vegetation scaling (set to 1 to disable)
V_MIN_CLIP = 0.1      # original: mean = 0.067, max = 0.207, min = 0.0079 (None to disable)

P_FAST_DIV_FACTOR = 7   # divide p_fast grid by this, 1 to use original

C_MIN_CAP = 0.01    # Min of C, to avoid dead areas when past
############################Parameters##############################

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