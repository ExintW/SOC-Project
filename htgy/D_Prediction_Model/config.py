import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paths import Paths 

################################ Run Config ################################
INIT_YEAR = 2007    # Year of initial SOC data (or first year of future for SKIP_TO_FUTURE)
END_YEAR = None     # End year of present simulation
FUTURE_YEAR = None  # End year of future simulation (Future year starts at END_YEAR + 1)
PAST_YEAR = 1950    # End year of reverse simulation

RUN_FROM_EQUIL = True      # If True, past run from equilibrium state instead of INIT_YEAR data, if False and USE_PAST_EQUIL is True, ALWAYS_USE_PAST must be True
EQUIL_YEAR = 2014           # Year to use as equilibrium state for reverse simulation

CLEAN_OUTDIR = True          # If True, clean output directory before running

FUTURE_INIT_FILE = Paths.OUTPUT_DIR / "Data" / "SOC_Present 7" / "SOC_terms_2024_12_River.parquet"  # File containing the parquet data for 2024/12
SKIP_TO_FUTURE = False      # If True, skip to future initial data directly (only if FUTURE_INIT_FILE is set)

VALIDATE_PAST = False        # If True, validate past SOC against simulated data

SAVE_NC = False              # If True, additionally save output as NetCDF files
USE_PARQUET = True           # Use parquet to store output, if false, use csv instead

PRINT_MAX = False            # DEBUG: print all max values for each timestep
PRINT_ALL = False            # DEBUG: Print all values for each timestep
################################ Simulation/Model Config ##########################
C_INIT_FACTOR = 1           # Adjust initial value of SOC (Set to 1 to use original)
C_INIT_CAP = 80             # Cap initial SOC to this value (0 to disable)
P_FAST_DIV_FACTOR = 10      # divide p_fast grid by this, 1 to use original   
SOC_PAST_FACTOR = 1         # Adjust past year SOC values (Set to 1 to use original)
C_MIN_CAP = 0.001            # Min of C, to avoid dead areas when past
C_FAST_MAX = 7 
C_SLOW_MAX = 1e9 

USE_GAUSSIAN_BLUR = True    # If True, apply Gaussian blur to past SOC data for smoother prior
SIGMA = 10                  # Strength of the gaussian blur

USE_PAST_LAI_TREND = True   # If True, use past LAI trend for regularization

V_FACTOR = 8                # for vegetation scaling (set to 1 to disable)
V_MIN_CLIP = 0              # clip V (None to disable)
V_FAST_PROP = 0.8           # for vegetation input proportion
A_MAX = 1                   # clip A

ALPHA = 0.20                # for humification -> % minerized C fast that becomes C slow  (0 to disable)

################################ Regularization Config ##########################
USE_TIKHONOV = True         # Enable L2 Regularization
PAST_KNOWN = 1980               # Year of 1 known SOC data in the past
REG_FREQ = 5                # Frequency (in years) to apply spatial regularization

USE_PAST_EQUIL = True       # If True, use past equilibrium SOC for regularization
ALWAYS_USE_PAST = False     # if True, always use PAST_KNOWN as prior knowledge (USE_PAST_EQUIL needs to be True)

L_FAST_MIN = 0.1            # Regularization Term
L_SLOW_MIN = 0.1            # Regularization Term

PLOT_PRIOR = False          # Plot the piror SOC of that time step when doing reg

# Spatial Regularization
USE_SPATIAL_REG = False     # Unequal regularization
USE_K_FOR_SPATIAL = False   # If False, use A for spatial. K for spatial uses different lambda for C fast and slow
ADD_V_IN_SPATIAL = False    # Add more REG in low V areas
REG_CONST_BASE = 0.125
REG_ALPHA = 5                  # Adjust the impact of K or A on REG
REG_BETA = 5                    # Adjust the impact of V on REG

REG_CONST = 1                   # Not using this if spatial reg is true

# Using 1 known past year for reg
USE_PAST_EQUIL = True           # if True, past will use PAST soc as prior knowledge if cur year is closer to PAST
USE_PAST_EQUIL_AVG = False      # Use the avg of PAST_KNOWN and equil year as prior
USE_PAST_EQUIL_PREV_AVG = False  # Use the avg of PAST_KNOWN, equil year, and previous month as prior
USE_DYNAMIC_AVG = True          # Use weighted avg of PAST_KNOWN and EQUIL between PAST_KNOWN and EQUIL
USE_PRIOR_PREV_AVG = False      # Use the avg of prior year (EQUIL or 1980) and previous timestep as prior

################################ Region Specific Config ################################
BORDER_SHP = Paths.DATA_DIR / "Loess_Plateau_vector_border.shp"                 # Shapefile for the border
INIT_SOC_CSV = Paths.PROCESSED_DIR / "Resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"   # CSV containing init values for SOC along with DEM, k1, k2, etc.
DAM_CSV = Paths.PROCESSED_DIR / "htgy_Dam_with_matched_points.csv"              # CSV containing dam info with matched Lon/Lat
FAST_SLOW_RATIO_CSV = Paths.DATA_DIR / "Fast_Slow_SOC_Proportion.csv"           # CSV containing fast/slow SOC proportions for each region
PAST_SOC_NPZ = Paths.PROCESSED_DIR / "soc_resampled_1980_matrix.npz"            # Numpy npz file containing SOC grid for 1 past year data
SMALL_RIVER_BASIN_SHP = Paths.DATA_DIR / "River_Basin" / "htgy_River_Basin.shp"   # Shapefile for small river basins
LARGE_RIVER_BASIN_SHP = Paths.DATA_DIR / "River_Basin" / "94_area.shp"          # Shapefile for large river basins
RIVERS_SHP = Paths.DATA_DIR / "China_River" / "ChinaRiver_main.shp"
LAI_PAST_FILE = Paths.PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_1950-2000.nc"  # NetCDF file containing resampled LAI data for past years
LS_FILE = Paths.PROCESSED_DIR / "LS_factor.npy"                                 # file name for precomputed LS factor grid (this file can be generated if not exist)
DEM_FILE_NAME = "htgyDEM.tif"                                                   # DEM file used for LS factor calculation
CMIP6_PR_FILE = Paths.PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_pr_points_2015-2100_585.nc" # NetCDF file containing resampled precipitation data for CMIP6
ERA5_PR_DIR = Paths.PROCESSED_DIR / "ERA5_Data_Monthly_Resampled"               # Directory containing resampled ERA5 precipitation data for present years
LOW_POINT_CSV = Paths.PROCESSED_DIR / "Low_Point_Summary.csv"                   # CSV containing low point info

CMIP_START = 1950                  # Start year for CMIP6 LAI data
DESIRED_CRS = "EPSG:4326"          # Desired coordinate reference system for all spatial data
DEM_RESOLUTION = 30                # Resolution of DEM data in meters  
GRID_RESOLUTION = 1000             # Resolution of model grid in meters

# Constants
BULK_DENSITY = 1300                 # Convert dam capacity from 10,000 m³ to tons using a bulk density of 1300 t/m³.
CELL_AREA_HA = 100.0                # 1 km² = 100 ha
DEPTH = 0.2                         # Depth of soil layer in meters

# Column names for INIT_SOC_CSV
LON_COL = "LON"
LAT_COL = "LAT"
SOC_COL = "ORGA"                       # Initial SOC concentration (g/kg)
DEM_COL = "htgy_DEM"                   # DEM (elevation)
LANDUSE_COL = "LANDUSE"                # Land use
REGION_COL = "Region"                  # Not used later
SLOPE_COL = "SLOPE"                    # Slope values
K1_COL = "SOC_k1_fast_pool (1/month)"  # Fast pool decay rate
K2_COL = "SOC_k2_slow_pool (1/month)"  # Slow pool decay rate
SAND_COL = "SAND"                      # Sand fraction
SILT_COL = "SILT"                      # Silt fraction
CLAY_COL = "CLAY"                      # Clay fraction

# Column names for DAM_CSV
DAM_YEAR = "year"
DAM_STORAGE = "total_stor"
DAM_DEPOSITION = "deposition"
DAM_CAPACITY_REMAINED = "capacity_remained"

# Column names for LAI_PAST_FILE
LAI_LON = 'lon'
LAI_LAT = 'lat'
LAI_VAR = 'lai'

CMIP6_LAI_SEGMENTS = [      # Specify segments of CMIP6 LAI data files covering different time periods
    {
        "start_year": 1950,
        "end_year": 2000,
        "cmip_start": 1950,
        "relpath": Paths.PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_1950-2000.nc",
    },
    {
        "start_year": 2001,
        "end_year": 2014,
        "cmip_start": 2001,
        "relpath": Paths.PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2001-2014.nc",
    },
    {
        "start_year": 2015,
        "end_year": None,  # open-ended
        "cmip_start": 2015,
        "relpath": Paths.PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_585.nc",
    },
]

# Column names for CMIP6_LAI and CMIP6_PR
CMIP_LON = 'lon'
CMIP_LAT = 'lat'
CMIP_LAI = 'lai'
CMIP_PR  = 'pr'
CMIP_PR_CONV_FACTOR = 30 * 86400  # Convert from kg m^-2 s^-1 to mm/month

# Column names for ERA5_PR
ERA5_LON = 'longitude'
ERA5_LAT = 'latitude'
ERA5_PR  = 'tp'      
ERA5_PR_CONV_FACTOR = 30 * 1000  # Convert from m/month to mm/month

def get_param_log():
        return f"""\
############################ Parameters ##############################
C_INIT_CAP = {C_INIT_CAP}
C_INIT_FACTOR = {C_INIT_FACTOR}
SOC_PAST_FACTOR = {SOC_PAST_FACTOR}
USE_TIKHONOV = {USE_TIKHONOV}
REG_CONST = {REG_CONST}
USE_SPATIAL_REG = {USE_SPATIAL_REG}
REG_CONST_BASE = {REG_CONST_BASE}
REG_ALPHA = {REG_ALPHA}
USE_K_FOR_SPATIAL = {USE_K_FOR_SPATIAL}
ADD_V_IN_SPATIAL = {ADD_V_IN_SPATIAL}
REG_BETA = {REG_BETA}
REG_FREQ = {REG_FREQ}
RUN_FROM_EQUIL = {RUN_FROM_EQUIL}
EQUIL_YEAR = {EQUIL_YEAR}
USE_PAST_EQUIL = {USE_PAST_EQUIL}
ALWAYS_USE_PAST = {ALWAYS_USE_PAST}
USE_PAST_EQUIL_AVG = {USE_PAST_EQUIL_AVG}
USE_PRIOR_PREV_AVG = {USE_PRIOR_PREV_AVG}
USE_PAST_EQUIL_PREV_AVG = {USE_PAST_EQUIL_PREV_AVG}
USE_PAST_LAI_TREND = {USE_PAST_LAI_TREND}
USE_GAUSSIAN_BLUR = {USE_GAUSSIAN_BLUR}
SIGMA = {SIGMA}
USE_DYNAMIC_AVG={USE_DYNAMIC_AVG}
ALPHA = {ALPHA}
A_MAX = {A_MAX}
L_FAST_MIN = {L_FAST_MIN}
L_SLOW_MIN = {L_SLOW_MIN}
V_FAST_PROP = {V_FAST_PROP}
V_FACTOR = {V_FACTOR}
V_MIN_CLIP = {V_MIN_CLIP}
P_FAST_DIV_FACTOR = {P_FAST_DIV_FACTOR}
C_MIN_CAP = {C_MIN_CAP}
C_FAST_MAX = {C_FAST_MAX}
C_SLOW_MAX = {C_SLOW_MAX}
######################################################################
"""