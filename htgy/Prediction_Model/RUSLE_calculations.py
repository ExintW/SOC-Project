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