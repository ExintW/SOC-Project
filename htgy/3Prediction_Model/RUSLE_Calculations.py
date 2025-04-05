import numpy as np
from globalss import *

# =============================================================================
# RUSLE COMPONENTS (MONTHLY)
# =============================================================================
def calculate_r_factor_monthly(rain_month_mm):
    """
    Compute the monthly rainfall erosivity factor (R factor).

    For the Loess Plateau, studies (e.g., Zhao et al. 2012) suggest a coefficient
    about 4 times higher than the standard value, so:

        R = 6.94 * rain_month_mm

    This adjustment yields soil loss values closer to observed rates (~1000 t/km²/year).
    """
    
    # return 6.94 * rain_month_mm
    return 0.739 * (rain_month_mm ** 1.56)

def calculate_r_factor_annually(rain_year_mm):
    """
    Compute R by using the Modified Fournier Index (MFI)
    
        MFI = Sum_{i=1}^{12}({P_i^2}/P) ; P = Annual tp
    
    Then use the experience formula:

        R = 1.735 * 10^{1.5 * log_10(MFI)}
    """
    annual_tp = np.sum(rain_year_mm, axis=0)
    # MFI = np.sum(rain_year**2, axis=0) / annual_tp
    # R = 1.735 * (10 ** (1.5 * np.log10(MFI)))
    
    R = 0.0534 * (annual_tp ** 1.6548)
    """
    Using regression formula:
    
        R = 587.8 - 1.219P + 0.004105P^2
    """
    # annual_tp = np.sum(rain_year_mm, axis=0)
    # R = 587.8 - 1.219 * annual_tp + 0.004105 * annual_tp**2
    
    return R

def get_montly_r_factor(R_annual, rain_month_mm, rain_year_mm):
    """
    Compute montly R factor using the ratio of montly precipitation
        
        R_i = R_annual * {P_i^2} / {P_annual^2}
    """
    annual_tp = np.sum(rain_year_mm, axis=0)
    R_month = R_annual * ((rain_month_mm ** 2) / (annual_tp ** 2))
    
    return R_month


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

def calculate_k_factor(silt, sand, clay, soc, landuse):
    """
    Using Wischmeier & Smith (1978) equation
        100K = 2.1e-4 * M^1.14(12 - OM) + 3.25(s - 2) + 2.5(p - 3)
    """
    s_values = {    # Structure Code = 1~4
        "sloping cropland": 4,
        "forestland": 1,
        "grassland": 2,
        "not used": 3,
        "terrace": 2,
        "dam field": 4
    }   
    p_values = {    # Permeability Code = 1~6
        "sloping cropland": 6,
        "forestland": 1,
        "grassland": 3,
        "not used": 5,
        "terrace": 3,
        "dam field": 4
    }  
 
    M = (silt + sand) * (100 - clay)
    K = 2.1e-4 * (M**1.14) * (12 - soc) + 3.25*(s_values.get(str(landuse).lower(), 2) - 2) + 2.5*(p_values.get(str(landuse).lower(), 3) - 3)
    # print(f'# of sloping cropland: {np.sum(landuse == 'sloping cropland')}')
    # print(f'# of forestland = {np.sum(landuse == 'forestland')}')
    # print(f'# of grassland = {np.sum(landuse == 'grassland')}')
    # print(f'# of not used = {np.sum(landuse == 'not used')}')
    # print(f'# of terrace = {np.sum(landuse == 'terrace')}')
    # print(f'# of dam field = {np.sum(landuse == 'dam field')}')
    if np.any(K < 0):
        print(f"Warning: negative values found for K factor!")
        # print(f"Negative K values = {K[K < 0]}")
    return K / 100