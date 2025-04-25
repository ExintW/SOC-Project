import numpy as np
import sys
import os
from whitebox.whitebox_tools import WhiteboxTools
import rasterio
from globalss import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import * 

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
    # MFI = np.sum(rain_year_mm**2, axis=0) / annual_tp
    # R = 1.735 * (10 ** (1.5 * np.log10(MFI)))
    
    # R = 0.0534 * (annual_tp ** 1.6548)
    """
    Using regression formula from Renard and Freimund (1994):
        P = total annual precipitation
        R = 0.0483 * P^1.61, if P <= 850mm
        R = 587.8 - 1.219P + 0.004105P^2, if P > 850mm
    """
    b = 1.61 # 1.61
    c = 2
    if np.mean(annual_tp) <= 850:
        R = 0.0483 * (annual_tp ** b)
    else:
        R = 587.8 - 1.219 * annual_tp + 0.004105 * annual_tp**2
    
    """
    https://doi.org/10.11821/dlxb201509012
    Angulo-Martínez M, Beguería S.
    Estimating rainfall erosivity from daily precipitation records: A comparison among methods using data from the Ebro Basin (NE Spain)

    . Journal of Hydrology, 2009, 379(1/2): 111-121.
    """
    # exponent = 1.5 * np.log10((rain_year_mm ** 2) / annual_tp) - 0.8188
    # R = np.sum(1.735 * (10 ** exponent), axis=0)  # sum over 12 months
    
    """
    Zhou et al.(1995) as cited in Li et al.(2014)
    """
    # R = np.sum(-1.15527 + 1.792 * rain_year_mm, axis=0)
    
    return R / c

def get_montly_r_factor(R_annual, rain_month_mm, rain_year_mm):
    """
    Compute montly R factor using the ratio of montly precipitation
        
        R_i = R_annual * {P_i^2} / {P_annual^2}
    """
    annual_tp = np.sum(rain_year_mm, axis=0)
    R_month = R_annual * ((rain_month_mm) / (annual_tp))
    # R_month = R_annual * ((rain_month_mm ** 2) / (annual_tp ** 2))
    
    return R_month


def calculate_ls_factor(slope, dem, slope_length=1000):
    """
    Compute LS factor from slope (degrees).
    This is a simplified formula; in real RUSLE, LS depends on slope length, slope steepness, etc.
    """
    # slope_rad = np.deg2rad(slope)
    # return ((slope_length / 22.13) ** 0.4) * ((np.sin(slope_rad) / 0.0896) ** 1.3)
    
    """
    Compute LS factor using simplified Moore & Burch (1986) method
    """
    # dzdx = (np.roll(dem, -1, axis=1) - np.roll(dem, 1, axis=1)) / (2 * slope_length)
    # dzdy = (np.roll(dem, -1, axis=0) - np.roll(dem, 1, axis=0)) / (2 * slope_length)
    # slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))

    # # 简化的 flow accumulation，用 slope 指代（实际建议用 D8 流向算法）
    # flow_acc = np.maximum(1, np.exp(slope_rad * 5))

    # LS = ((flow_acc * slope_length) / 22.13)**0.4 * (np.sin(slope_rad) / 0.0896)**1.3
    # return LS
    
    LS_file = PROCESSED_DIR / "LS_factor.npy"
    
    """
    Compute LS factor using whitebox and Moore & Burch (1986) method
    """
    if LS_file.exists():
        print("Loading precomputed LS_factor...")
        LS = np.load(LS_file)
    else:
        print("Computing LS factor using whitebox...")
        wbt = WhiteboxTools()
        wbt.verbose = False
        wbt.work_dir = str(DATA_DIR) # DEM directory

        dem_tif = "htgyDEM.tif"  # 用你的DEM文件替代
        flow_dir = "flow_dir.tif"
        flow_acc = "flow_acc.tif"
        slope = "slope.tif"

        # 步骤1：计算坡度（以弧度为单位）
        wbt.run_tool("slope", [
            f"--dem={dem_tif}",
            f"--output={slope}",
            "--zfactor=1.0",
            "--units=degrees",
            "--cores=8"
        ])

        # 步骤2：计算 D8 流向
        wbt.run_tool("d8_pointer", [
            f"--dem={dem_tif}",
            f"--output={flow_dir}",
            "--esri_pntr",
            "--cores=8"
        ])


        # 步骤3：计算累积汇流
        wbt.run_tool("d8_flow_accumulation", [
            f"--dem={dem_tif}",
            f"--output={flow_acc}",
            "--out_type=cells",
            "--cores=8"
        ])

        # 步骤4：读取flow_acc和slope计算LS
        with rasterio.open(DATA_DIR / flow_acc) as fac_src, rasterio.open(DATA_DIR / slope) as slope_src:
            fac = fac_src.read(1).astype(np.float32)
            slp = slope_src.read(1).astype(np.float32)

        # Moore & Burch (1986) 公式计算 LS
        cell_size = 30  # 你的 DEM 分辨率
        fac = np.maximum(fac, 1)  # 避免0
        slope_rad = np.deg2rad(slp)  # slope.tif 是角度，需要转弧度
        LS = ((fac * cell_size) / 22.13)**0.5 * (np.sin(slope_rad) / 0.0896)**1.5
        
        np.save(LS_file, LS)
    
    return LS

def calculate_c_factor(lai):
    """Compute C factor from LAI: C = exp(-1.7 * LAI)."""
    a = -1.7   # -1.7
    return np.exp(a * lai)

def calculate_p_factor(landuse, slope):
    """Return P factor based on land use category."""
    # p_values = {
    #     "sloping cropland": 0.4,
    #     "forestland": 0.5,
    #     "grassland": 0.5,
    #     "not used": 0.5,
    #     "terrace": 0.1,
    #     "dam field": 0.05
    # }
    # p_values = {    # https://doi.org/10.11821/dlxb201509012
    #     "sloping cropland": 1.0,  # 无水土保持措施的坡耕地
    #     "forestland": 1.0,        # 林地，未实施特定水土保持措施
    #     "grassland": 1.0,         # 草地，未实施特定水土保持措施
    #     "not used": 1.0,          # 未利用地，未实施特定水土保持措施
    #     "terrace": 0.12,          # 水平梯田，减沙效益可达88%
    #     "dam field": 0.05         # 淤地坝地，减沙效益显著
    # }
    
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
        return None  # or raise ValueError if desired
    
    p_values = {    # 近20年黄土高原林地土壤侵蚀时空变化特征及其影响因素 
        "sloping cropland": get_p_factor(slope),
        "forestland": 1,
        "grassland": 1,
        "not used": 1,
        "terrace": 0.12,    # https://doi.org/10.11821/dlxb201509012
        "dam field": 0.05
    }
    
    return p_values.get(str(landuse).lower(), 1.0)

def calculate_k_factor(silt, sand, clay, soc, landuse):
    """
    Using Wischmeier & Smith (1978) equation
        100K = 2.1e-4 * M^1.14(12 - OM) + 3.25(s - 2) + 2.5(p - 3)
    """
    # s_values = {    # Structure Code = 1~4
    #     "sloping cropland": 4,
    #     "forestland": 2,
    #     "grassland": 3,
    #     "not used": 4,
    #     "terrace": 3,
    #     "dam field": 4
    # }   
    # p_values = {    # Permeability Code = 1~6
    #     "sloping cropland": 6,
    #     "forestland": 3,
    #     "grassland": 5,
    #     "not used": 6,
    #     "terrace": 5,
    #     "dam field": 5
    # }  
 
    # M = (silt + sand) * (100 - clay)
    # K = 2.1e-4 * (M**1.14) * (12 - soc) + 3.25*(s_values.get(str(landuse).lower(), 2) - 2) + 2.5*(p_values.get(str(landuse).lower(), 3) - 3)
    # if np.any(K < 0):
    #     print(f"Warning: negative values found for K factor!")
    #     # print(f"Negative K values = {K[K < 0]}")
    # return K / 100  
    
    #return 0.03834  # Source: https://doi.org/10.57760/sciencedb.07135， 10.12041/geodata.201703065582271.ver1.db

    """
    EPIC Model (Williams, 1995)
    Williams and Renard (1983) as cited in Chen et al. (2011)
    """
    # Avoid division by zero
    # total = silt + clay
    # total[total == 0] = 1e-6

    # # Organic carbon factor
    # oc = soc / 10  # convert to percentage
    # oc_factor = (1 - 0.25 * oc) / (oc + np.exp(3.72 - 2.95 * oc))

    # # Texture-related terms
    # texture_term = (0.2 + (0.3 * np.exp(-0.0256 * sand * (1 - silt / 100)))) * \
    #                ((silt / total) ** 0.3)

    # # Final K factor
    # k_factor = texture_term * oc_factor
    
    """
    EPIC from https://doi.org/10.11821/dlxb201509012 
    基于土壤侵蚀控制度的黄土高原水土流失治理潜力研究
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
    return k_factor