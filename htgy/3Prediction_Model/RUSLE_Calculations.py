import numpy as np
from globalss import *

# =============================================================================
# RUSLE COMPONENTS (MONTHLY)
# =============================================================================
def calculate_r_factor_monthly(rain_month_mm):
    """Compute R factor from monthly precipitation: R = 6.94 * rain_month_mm."""
    return 6.94 * rain_month_mm

def calculate_ls_factor(slope, slope_length=1000):
    """
    Compute LS factor from slope (degrees).
    This is a simplified formula; in real RUSLE, LS depends on slope length, slope steepness, etc.
    """
    slope_rad = np.deg2rad(slope)
    return ((slope_length / 22.13) ** 0.4) * ((np.sin(slope_rad) / 0.0896) ** 1.3)

def calculate_c_factor(lai):
    """Compute C factor from LAI: C = exp(-1.7 * LAI)."""
    return np.exp(-1.7 * lai)

def calculate_p_factor(landuse):
    """Return P factor based on land use category."""
    p_values = {
        "sloping cropland": 0.4,
        "forestland": 0.5,
        "grassland": 0.5,
        "not used": 0.5,
        "terrace": 0.1,
        "dam field": 0.05
    }
    return p_values.get(str(landuse).lower(), 1.0)

