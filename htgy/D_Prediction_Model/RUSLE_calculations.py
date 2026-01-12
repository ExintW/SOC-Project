import os
import sys
import numpy as np
import rasterio

from whitebox.whitebox_tools import WhiteboxTools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import Paths 
from global_structs import MAP_STATS, INIT_VALUES
from config import *

def calculate_LS_factor(slope):
    """
    Compute LS factor using whitebox and Moore & Burch (1986) method
    Saves computed LS factor to LS_FILE for future use.
    """
    if LS_FILE.exists():
        print("Loading precomputed LS_factor...")
        return np.load(LS_FILE)
    else:
        print("Computing LS factor using whitebox...")
        wbt = WhiteboxTools()
        wbt.verbose = False
        wbt.workdir = str(Paths.DATA_DIR) # DEM directory
        
        dem_tif = DEM_FILE_NAME         # Input DEM file
        flow_dir = "flow_dir.tif"       # Temporary file for flow direction
        flow_acc = "flow_acc.tif"       # Temporary file for flow accumulation
        slope_tif = "slope.tif"         # Temporary file for slope
        
        # Calculate slope in degrees
        wbt.run_tool("slope", [
            f"--dem={dem_tif}",
            f"--output={slope}",
            "--zfactor=1.0",
            "--units=degrees",
            "--cores=8"
        ])

        # Calculate D8 flow direction
        wbt.run_tool("d8_pointer", [
            f"--dem={dem_tif}",
            f"--output={flow_dir}",
            "--esri_pntr",
            "--cores=8"
        ])
        
        # Calculate flow accumulation
        wbt.run_tool("d8_flow_accumulation", [
            f"--dem={dem_tif}",
            f"--output={flow_acc}",
            "--out_type=cells",
            "--cores=8"
        ])
        
        # Read flow accumulation and slope to compute LS
        with rasterio.open(Paths.DATA_DIR / flow_acc) as fac_src, rasterio.open(Paths.DATA_DIR / slope_tif) as slope_src:
            fac = fac_src.read(1).astype(np.float32)
            slp = slope_src.read(1).astype(np.float32)
            
        # Use Moore & Burch (1986) formula to compute LS
        cell_size = DEM_RESOLUTION      # DEM resolution in meters
        fac = np.maximum(fac, 1)        # Avoid division by zero
        slope_rad = np.deg2rad(slp)     # Convert slope from degrees to radians
        LS = ((fac * cell_size) / 22.13)**0.5 * (np.sin(slope_rad) / 0.0896)**1.5
        
        np.save(LS_FILE, LS)
    
    return LS

def calculate_p_factor(landuse, slope):
    """
    Compute P factor based on land use and slope
    """
    
    def get_p_factor(slope):
        cropland_map = {
            (0, 5): 0.1,
            (5, 10): 0.221,
            (10, 15): 0.305,
            (15, 20): 0.575,
            (20, 25): 0.705,
            (25, float('inf')): 0.8 
        }
        for slope_range, p_value in cropland_map.items():
            if slope_range[0] <= slope < slope_range[1]:
                return p_value
        return None  
    
    p_values = { 
        "sloping cropland": get_p_factor(slope),
        "forestland": 1,
        "grassland": 1,
        "not used": 1,
        "terrace": 0.12,    # https://doi.org/10.11821/dlxb201509012
        "dam field": 0.05
    }
    
    return p_values.get(str(landuse).lower(), 1.0)

def calculate_r_factor_annually(rain_year_mm):
    annual_tp = np.sum(rain_year_mm, axis=0)
    
    """
    Using regression formula from Renard and Freimund (1994):
        P = total annual precipitation
        R = 0.0483 * P^1.61, if P <= 850mm
        R = 587.8 - 1.219P + 0.004105P^2, if P > 850mm
    """
    if np.mean(annual_tp) <= 850:
        R = 0.0483 * (annual_tp ** 1.78)
    else:
        R = 587.8 - 1.219 * annual_tp + 0.004105 * annual_tp**2
    
    return R / 5.5

def calculate_c_factor(lai):
    """Compute C factor from LAI: C = exp(-1.7 * LAI)."""
    # https://doi.org/10.3390/rs15112868
    C = -0.177 * np.log(lai) + 0.184
    return np.clip(C, a_min=1e-6, a_max=None)

def get_monthly_r_factor(R_annual, rain_month_mm, rain_year_mm):
    """
    Compute montly R factor using the ratio of montly precipitation
        
        R_i = R_annual * {P_i^2} / {P_annual^2}
    """
    annual_tp = np.sum(rain_year_mm, axis=0)
    R_month = R_annual * ((rain_month_mm) / (annual_tp))
    
    return R_month

def calculate_k_factor(silt, sand, clay, soc, landuse):
    """
    EPIC from https://doi.org/10.11821/dlxb201509012 
    """
    # Avoid division by zero
    total = silt + clay
    total[total == 0] = 1e-9
    SN_1 = 1 - (sand / 100)

    # Organic carbon factor
    oc = soc / 10  # convert to percentage
    oc_factor = 1 - ((0.25 * oc) / (oc + np.exp(3.72 - 2.95 * oc)))

    # Texture-related terms
    texture_term = 0.2 + (0.3 * np.exp(-0.0256 * sand * (1 - silt / 100))) * \
                   ((silt / total) ** 0.3)
    
    SN_term = 1 - ((0.7 * SN_1) / (SN_1 + np.exp(-5.51 + 22.9 * SN_1)))

    # Final K factor
    k_factor = 0.1317 * texture_term * oc_factor * SN_term
    # k_factor = texture_term * oc_factor * SN_term
    return np.clip(k_factor, a_min=1e-6, a_max=None)

def vegetation_input(LAI):
    """
    Compute vegetation input based on LAI using an empirical formula.
    """
    LAI_safe = np.maximum(LAI, 1e-6)
    return 0.1587 * np.log(LAI_safe) + 0.1331
