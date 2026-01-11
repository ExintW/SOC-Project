import os
import sys
import netCDF4 as nc
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import Paths 
from global_structs import MAP_STATS, INIT_VALUES
from config import *
from utils import convert_soil_to_soc_loss, resolve_cmip6_lai_segment, init_dams, create_grid_from_points, convert_soil_loss_to_soc_loss_monthly
from RUSLE_calculations import calculate_r_factor_annually, calculate_c_factor, get_monthly_r_factor, calculate_k_factor, vegetation_input 
from SOC_Dynamic_Model import soc_dynamic_model

def run_simulation_year(year, past=False, future=False):
    print(f"Running Simulation year {year}...")
    
    # Initialize dams for this year
    init_dams(year)
    
    # Get pr file paths
    if future:
        pr_file = CMIP6_PR_FILE
    else:
        pr_file = ERA5_PR_DIR / f"resampled_{year}.nc"
    
    # Get lai file paths
    lai_file, cmip_start = resolve_cmip6_lai_segment(year, CMIP6_LAI_SEGMENTS)
    
    with nc.Dataset(pr_file) as ds_pr, nc.Dataset(lai_file) as ds_lai:
        n_time = 12
        
        # Load LAI and PR data for the year
        lon_lai = ds_lai.variables[CMIP_LON][:]  # Adjusted variable name if needed
        lat_lai = ds_lai.variables[CMIP_LAT][:]
        lai_data = ds_lai.variables[CMIP_LAI][:]  # shape: (12, n_points)
            
        if future:
            lon_pr = ds_pr.variables[CMIP_LON][:]
            lat_pr = ds_pr.variables[CMIP_LAT][:]
            pr_data = ds_pr.variables[CMIP_PR][:]
            tp_data_mm = pr_data * CMIP_PR_CONV_FACTOR  # convert to mm/month

            start_idx = (year - cmip_start) * n_time
            end_idx = start_idx + n_time
        
            if start_idx < 0 or end_idx > tp_data_mm.shape[0]:
                raise ValueError(f"No CMIP6 data for year {year} (idx {start_idx}:{end_idx})")
            
            # slice out the 12 months, no summing
            tp_data_mm = tp_data_mm[start_idx:end_idx, :]  # shape = (12, n_points)
        else:
            lon_pr = ds_pr.variables[ERA5_LON][:]
            lat_pr = ds_pr.variables[ERA5_LAT][:]
            pr_data = ds_pr.variables[ERA5_PR][:]         # shape: (time, n_points), in m/month
            tp_data_mm = pr_data * ERA5_PR_CONV_FACTOR  # convert to mm/month
        
        R_annual = calculate_r_factor_annually(tp_data_mm)
        
        E_month_avg_list = []   # for calculating annual mean for validation
        C_month_list = []
        if past:
            time_range = range(n_time-1, -1, -1)    # 11 -> 0
        else:
            time_range = range(n_time)
            
        for month_idx in time_range:
            time_month = time.time()
            
            idx = (year - cmip_start) * n_time + month_idx  # flat index relative to cmip_start
            if idx < 0 or idx >= lai_data.shape[0]:
                continue
            
            print(f"\n=======================================================================")
            print(f"                          Year {year} Month {month_idx+1}")
            print(f"=======================================================================\n")
            
            # Load LAI data and compute C factor
            lai_1d = lai_data[idx, :]
            lai_2d = create_grid_from_points(lon_lai, lat_lai, lai_1d, MAP_STATS.grid_x, MAP_STATS.grid_y)
            lai_2d = np.nan_to_num(lai_2d, nan=np.nanmean(lai_2d))
            lai_2d[~MAP_STATS.border_mask] = np.nan
            
            C_factor_2D = calculate_c_factor(lai_2d)
            C_factor_2D[~MAP_STATS.border_mask] = np.nan
            C_month_list.append(np.nanmean(C_factor_2D))
            
            # Load PR data and compute R factor
            if future:
                tp_1d_mm = tp_data_mm[idx, :]
            else:
                tp_1d_mm = tp_data_mm[month_idx, :]
            R_month = get_monthly_r_factor(R_annual, tp_1d_mm, tp_data_mm)
            R_month = create_grid_from_points(lon_pr, lat_pr, R_month, MAP_STATS.grid_x, MAP_STATS.grid_y)
            R_month = np.nan_to_num(R_month, nan=np.nanmean(R_month))
            R_month[~MAP_STATS.border_mask] = np.nan
            
            # Calculate Monthly K Factor
            K_month = calculate_k_factor(INIT_VALUES.SILT, INIT_VALUES.SAND, INIT_VALUES.CLAY, (MAP_STATS.C_fast_current + MAP_STATS.C_slow_current), INIT_VALUES.LANDUSE)
            K_month = np.nan_to_num(K_month, nan=np.nanmean(K_month))
            K_month[~MAP_STATS.border_mask] = np.nan

            E_t_ha_month = R_month * K_month * INIT_VALUES.LS_FACTOR * C_factor_2D * INIT_VALUES.P_FACTOR
            E_t_ha_month[~MAP_STATS.border_mask] = np.nan
            E_tcell_month = E_t_ha_month * CELL_AREA_HA
            E_month_avg_list.append(np.nanmean(E_t_ha_month))
            
            SOC_loss_g_kg_month = convert_soil_loss_to_soc_loss_monthly(
                E_t_ha_month, (MAP_STATS.C_fast_current + MAP_STATS.C_slow_current)
            )
            
            A = convert_soil_to_soc_loss(E_t_ha_month)
            V = vegetation_input(lai_2d)
            
            soc_time = time.time()
            # Run SOC Dynamic Model to update SOC pools
            LAI_avg = np.nanmean(lai_2d)
            soc_dynamic_model(E_tcell_month, A, V, month_idx, year, past, LAI_avg)
            