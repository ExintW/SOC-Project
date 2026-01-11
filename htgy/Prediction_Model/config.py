import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paths import Paths 

################################ Run Config ################################
INIT_YEAR = 2007    # Year of initial SOC data (or first year of future for SKIP_TO_FUTURE)
END_YEAR = 2024     # End year of present simulation
FUTURE_YEAR = 2100  # End year of future simulation (Future year starts at END_YEAR + 1)
PAST_YEAR = 1950    # End year of reverse simulation

RUN_FROM_EQUIL = True       # If True, past run from equilibrium state instead of INIT_YEAR data
EQUIL_YEAR = 2014           # Year to use as equilibrium state for reverse simulation

CLEAN_OUTDIR = True          # If True, clean output directory before running

FUTURE_INIT_FILE = Paths.OUTPUT_DIR / "Data" / "SOC_Present 7" / "SOC_terms_2024_12_River.parquet"  # File containing the parquet data for 2024/12
SKIP_TO_FUTURE = False      # If True, skip to future initial data directly (only if FUTURE_INIT_FILE is set)

VALIDATE_PAST = False        # If True, validate past SOC against simulated data

SAVE_NC = False              # If True, additionally save output as NetCDF files

################################ Simulation/Model Config ##########################
C_INIT_FACTOR = 1           # Adjust initial value of SOC (Set to 1 to use original)
C_INIT_CAP = 80             # Cap initial SOC to this value (0 to disable)
P_FAST_DIV_FACTOR = 10      # divide p_fast grid by this, 1 to use original   
SOC_PAST_FACTOR = 1         # Adjust past year SOC values (Set to 1 to use original)

USE_GAUSSIAN_BLUR = True    # If True, apply Gaussian blur to past SOC data for smoother prior
SIGMA = 10                  # Strength of the gaussian blur

USE_PAST_LAI_TREND = True   # If True, use past LAI trend for regularization

################################ Regularization Config ##########################
REG_YEAR = 1980             # Year used for regularization
REG_FREQ = 5                # Frequency (in years) to apply spatial regularization

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

CMIP_START = 1950                  # Start year for CMIP6 LAI data
DESIRED_CRS = "EPSG:4326"          # Desired coordinate reference system for all spatial data
DEM_RESOLUTION = 30                # Resolution of DEM data in meters  
GRID_RESOLUTION = 1000             # Resolution of model grid in meters

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