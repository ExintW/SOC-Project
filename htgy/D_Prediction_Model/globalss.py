import numpy as np

USE_PARQUET = True  # Save output df as parquet instead of csv
CLEAN_OUTDIR = True
SAVE_NC = False

desired_crs = "EPSG:4326"
#desired_crs = "EPSG:3857"
BULK_DENSITY = 1300  # Convert dam capacity from 10,000 m³ to tons using a bulk density of 1300 t/m³.
DEPTH = 0.2
CELL_AREA_HA = 100.0  # 1 km² = 100 ha
M_soil = 2.6e8  # total soil mass per cell (kg)
INIT_YEAR = 2007
PRESENT_YEAR = 2025
# global_timestep = 0
USE_CMIP6 = True            # Use CMIP6 lai for present and past simulations (Uses ERA5 if False)

############################ UNet Hyperparameters ##############################
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
PRINT_FREQ = 10

############################ Parameters ##############################
C_INIT_CAP = 80
C_INIT_FACTOR = 1

USE_TIKHONOV = False         # If this is True, RUN_FROM_EQUIL also has to be True
REG_CONST = 0.275           # Not using this if spatial reg is true
USE_SPATIAL_REG = True      
REG_CONST_BASE = 0.01      # 0.2 # 0.25
REG_ALPHA = 20              # 1 # 10
USE_K_FOR_SPATIAL = True   # If False, use A for spatial. K for spatial uses different lambda for C fast and slow

RUN_FROM_EQUIL = True       # if True, past will start from end_year
EQUIL_YEAR = 2009           # Make sure to set end_year to this if run from equil
USE_1980_EQUIL = True       # if True, past will use 1980 soc as prior knowledge if cur year is closer to 1980
ALWAYS_USE_1980 = True      # if True, always use 1980 as prior knowledge

FAST_DAMP_START = 0.5       # only damp if any of C_fast_current is > this value
LAMBDA_FAST = 0.99          # for damping, set to 0 to disable
FAST_DAMP_THRESH = 1e9      # 0.4  # if diff > this value, then do damping (0 to damp all, inf to disable damp)
LAMBDA_SLOW = 0             # for damping,   set to 0 to disable

ALPHA = 0.20                # for humification -> % minerized C fast that becomes C slow  (0 to disable)

A_MAX = 1 # 0.1
D_MAX = 1e9

L_FAST_MIN = 0 # 0.7
L_SLOW_MIN = 0 # 0.95

K_SLOW_MAX = 1e9 # 0.08

V_FAST_PROP = 0.8           # for vegetation input proportion
V_FACTOR = 7                # for vegetation scaling (set to 1 to disable)
V_MIN_CLIP = 0 # 0.01           # original: mean = 0.067, max = 0.207, min = 0.0079 (None to disable)
V_SCALING_FACTOR = 0      # for additional V gain that is scaling with SOC: V = V + V_SCALING_FACTOR * SOC, 0 to disable

P_FAST_DIV_FACTOR = 10      # divide p_fast grid by this, 1 to use original

C_MIN_CAP = 0.001            # Min of C, to avoid dead areas when past
C_FAST_MAX = 1e9 # 2
C_SLOW_MAX = 1e9 # 10
######################################################################

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
    SOC_1980_FAST = None
    SOC_1980_SLOW = None
    
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
    
    C_fast_prev = None
    C_slow_prev = None

    C_fast_equil_list = []
    C_slow_equil_list = []

    @classmethod
    def reset(cls):
        for key in list(cls.__dict__):
            if not key.startswith('__') and not callable(getattr(cls, key)):
                setattr(cls, key, None)