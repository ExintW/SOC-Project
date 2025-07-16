import numpy as np

USE_PARQUET = True  # Save output df as parquet instead of csv
CLEAN_OUTDIR = True
SAVE_NC = True

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
PRINT_FREQ = 1

USE_UNET = False

############################ Parameters ##############################
C_INIT_CAP = 80
C_INIT_FACTOR = 1

#------------------------Regularization------------------------#
USE_TIKHONOV = True             # If this is True, RUN_FROM_EQUIL or ALWAYS_USE_1980 also has to be True
REG_CONST = 0.275               # Not using this if spatial reg is true
USE_SPATIAL_REG = True      
REG_CONST_BASE = 0.15            # 0.2 # 0.25
REG_ALPHA = 15                  # Adjust the impact of K or A on REG
USE_K_FOR_SPATIAL = True        # If False, use A for spatial. K for spatial uses different lambda for C fast and slow
ADD_V_IN_SPATIAL = True         # Add more REG in low V areas
REG_BETA = 20                    # Adjust the impact of V on REG

REG_FREQ = 4                    # Apply regularization every REG_FREQ months

# Prior Knowledge related
RUN_FROM_EQUIL = True           # if True, past will start from end_year=EQUIL_YEAR
EQUIL_YEAR = 2009               # Make sure to set end_year to this if run from equil
USE_1980_EQUIL = True           # if True, past will use 1980 soc as prior knowledge if cur year is closer to 1980
ALWAYS_USE_1980 = True          # if True, always use 1980 as prior knowledge (USE_1980_EQUIL needs to be True)
# The following options should be mutually exclusive
USE_1980_EQUIL_AVG = False      # Use the avg of 1980 and equil year as prior
USE_PRIOR_PREV_AVG = True      # Use the avg of prior year (EQUIL or 1980) and previous timestep as prior
USE_1980_EQUIL_PREV_AVG = False  # Use the avg of 1980, equil year, and previous month as prior
#---------------------------------------------------------------#

#------------------------Damping------------------------#
FAST_DAMP_START = 1e9 # 0.5       # only damp if any of C_fast_current is > this value
LAMBDA_FAST = 0.99          # for damping, set to 0 to disable
FAST_DAMP_THRESH = 1e9      # 0.4  # if diff > this value, then do damping (0 to damp all, inf to disable damp)
LAMBDA_SLOW = 0             # for damping,   set to 0 to disable
#-------------------------------------------------------#

ALPHA = 0.20                # for humification -> % minerized C fast that becomes C slow  (0 to disable)

A_MAX = 0.1
D_MAX = 1e9

L_FAST_MIN = 0.1 # 0.7
L_SLOW_MIN = 0.1 # 0.95

K_SLOW_MAX = 1e9 # 0.08

V_FAST_PROP = 0.8           # for vegetation input proportion
V_FACTOR = 8                # for vegetation scaling (set to 1 to disable)
V_MIN_CLIP = 0 # 0.01           # original: mean = 0.067, max = 0.207, min = 0.0079 (None to disable)
V_SCALING_FACTOR = 0      # for additional V gain that is scaling with SOC: V = V + V_SCALING_FACTOR * SOC, 0 to disable

P_FAST_DIV_FACTOR = 10      # divide p_fast grid by this, 1 to use original

C_MIN_CAP = 0.001            # Min of C, to avoid dead areas when past
C_FAST_MAX = 7 # 2
C_SLOW_MAX = 1e9 # 10
######################################################################

def get_param_log():
        return f"""\
############################ Parameters ##############################
C_INIT_CAP = {C_INIT_CAP}
C_INIT_FACTOR = {C_INIT_FACTOR}
USE_TIKHONOV = {USE_TIKHONOV}
REG_CONST = {REG_CONST}
USE_SPATIAL_REG = {USE_SPATIAL_REG}
REG_CONST_BASE = {REG_CONST_BASE}
REG_ALPHA = {REG_ALPHA}
USE_K_FOR_SPATIAL = {USE_K_FOR_SPATIAL}
ADD_V_IN_SPATIAL = {ADD_V_IN_SPATIAL}
REG_BETA = {REG_BETA}
RUN_FROM_EQUIL = {RUN_FROM_EQUIL}
EQUIL_YEAR = {EQUIL_YEAR}
USE_1980_EQUIL = {USE_1980_EQUIL}
ALWAYS_USE_1980 = {ALWAYS_USE_1980}
FAST_DAMP_START = {FAST_DAMP_START}
LAMBDA_FAST = {LAMBDA_FAST}
FAST_DAMP_THRESH = {FAST_DAMP_THRESH}
LAMBDA_SLOW = {LAMBDA_SLOW}
ALPHA = {ALPHA}
A_MAX = {A_MAX}
D_MAX = {D_MAX}
L_FAST_MIN = {L_FAST_MIN}
L_SLOW_MIN = {L_SLOW_MIN}
K_SLOW_MAX = {K_SLOW_MAX}
V_FAST_PROP = {V_FAST_PROP}
V_FACTOR = {V_FACTOR}
V_MIN_CLIP = {V_MIN_CLIP}
V_SCALING_FACTOR = {V_SCALING_FACTOR}
P_FAST_DIV_FACTOR = {P_FAST_DIV_FACTOR}
C_MIN_CAP = {C_MIN_CAP}
C_FAST_MAX = {C_FAST_MAX}
C_SLOW_MAX = {C_SLOW_MAX}
######################################################################
"""

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
    UNet_Model = None
    
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
    
    REG_counter = REG_FREQ

    @classmethod
    def reset(cls):
        for key in list(cls.__dict__):
            if not key.startswith('__') and not callable(getattr(cls, key)):
                setattr(cls, key, None)