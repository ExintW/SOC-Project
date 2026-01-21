import os
import sys
import netCDF4 as nc
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import Paths 
from global_structs import MAP_STATS, INIT_VALUES
from config import *
from utils import convert_soil_to_soc_loss, resolve_cmip6_lai_segment, init_dams, create_grid_from_points, convert_soil_loss_to_soc_loss_monthly, plot_SOC_timestep
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
            LAI_avg = np.nanmean(lai_2d)
            # Run SOC Dynamic Model to update SOC pools
            dep_soc_fast, dep_soc_slow, lost_soc = soc_dynamic_model(E_tcell_month, A, V, month_idx, year, past, LAI_avg)
            print(f'SOC Dynamic Model took {time.time() - soc_time}')
            
            store_plot_output(year, month_idx, past, SOC_loss_g_kg_month, dep_soc_fast, dep_soc_slow, V, E_t_ha_month, C_factor_2D, K_month, R_month, lost_soc)
            
            print(f"Completed simulation for Year {year}, Month {month_idx+1}")
            print(f"This month took {time.time() - time_month} seconds")
        
        print(f'C factor annual avg = {np.nanmean(C_month_list)}')
        print(f"\nAnnual mean of E = {np.nanmean(E_month_avg_list)}\n")
        
        return
            
def store_plot_output(year, month_idx, past, SOC_loss_g_kg_month, dep_soc_fast, dep_soc_slow, V, E_t_ha_month, C_factor_2D, K_month, R_month, lost_soc):
    if year == EQUIL_YEAR and not past:
                MAP_STATS.C_fast_equil_list.append(MAP_STATS.C_fast_current)
                MAP_STATS.C_slow_equil_list.append(MAP_STATS.C_slow_current)
            
    if VALIDATE_PAST and year == PAST_KNOWN:
        MAP_STATS.C_total_Past_Valid_list.append(MAP_STATS.C_fast_current + MAP_STATS.C_slow_current)
    
    C_total = MAP_STATS.C_fast_current + MAP_STATS.C_slow_current
    mean_C_total = np.nanmean(C_total)
    max_C_total = np.nanmax(C_total)
    min_C_total = np.nanmin(C_total)
    
    # stash a copy of this month’s total‐C grid
    if past:
        MAP_STATS.total_C_matrix.insert(0, C_total.copy())
        MAP_STATS.C_fast_matrix.insert(0, MAP_STATS.C_fast_current.copy())
        MAP_STATS.C_slow_matrix.insert(0, MAP_STATS.C_slow_current.copy())
        MAP_STATS.active_dam_matrix.insert(0, MAP_STATS.active_dams.copy())
        MAP_STATS.full_dam_matrix.insert(0, MAP_STATS.full_dams.copy())
        MAP_STATS.dam_rem_cap_matrix.insert(0, MAP_STATS.dam_rem_cap.copy())
    else:
        MAP_STATS.total_C_matrix.append(C_total.copy())
        MAP_STATS.C_fast_matrix.append(MAP_STATS.C_fast_current.copy())
        MAP_STATS.C_slow_matrix.append(MAP_STATS.C_slow_current.copy())
        MAP_STATS.active_dam_matrix.append(MAP_STATS.active_dams.copy())
        MAP_STATS.full_dam_matrix.append(MAP_STATS.full_dams.copy())
        MAP_STATS.dam_rem_cap_matrix.append(MAP_STATS.dam_rem_cap.copy())
        
    # count and report cells where C_total > 40 and it's an active dam
    high_dam_mask = (C_total > 40) & MAP_STATS.active_dams
    count_high_dam = np.count_nonzero(high_dam_mask)
    print(f"Number of active dam cells with C_total > 40: {count_high_dam}")

    count_full_dam = np.count_nonzero(MAP_STATS.full_dams)
    print(f"Number of full dams: {count_full_dam}")
    
    dam_cap = np.nansum(MAP_STATS.dam_rem_cap, dtype=np.float64)
    print(f"Remaining dam capacity: {dam_cap} tons")

    print(f"Year {year} Month {month_idx + 1}: Total_SOC_mean: {mean_C_total:.2f}, "
            f"max: {max_C_total:.2f}, min: {min_C_total:.2f}")
    print(f"C_fast_mean: {np.nanmean(MAP_STATS.C_fast_current):.2f}, C_slow_mean: {np.nanmean(MAP_STATS.C_slow_current):.2f}")
    
    plot_SOC_timestep(year, month_idx)
    
    lat_grid, lon_grid = np.meshgrid(MAP_STATS.grid_y, MAP_STATS.grid_x, indexing='ij')

    lat_list =  lat_grid.ravel(order='C').tolist()
    lon_list =  lon_grid.ravel(order='C').tolist()
    landuse_list = INIT_VALUES.LANDUSE.astype(str).ravel(order='C').tolist()
    Region_list = INIT_VALUES.REGION.astype(str).ravel(order='C').tolist()
    Low_point_list = MAP_STATS.low_mask.astype(str).ravel('C').tolist()

    pf = MAP_STATS.p_fast_grid
    sign = 1 if past else -1
    
    # SOC groups
    C_fast_list  = MAP_STATS.C_fast_current .ravel('C').tolist()
    C_slow_list  = MAP_STATS.C_slow_current .ravel('C').tolist()
    lost_soc_list =  lost_soc     .ravel('C').tolist()
    C_total_list = C_total        .ravel('C').tolist()

    # Erosion
    erosion_fast_list = ( sign * SOC_loss_g_kg_month *  pf          ).ravel('C').tolist()
    erosion_slow_list = ( sign * SOC_loss_g_kg_month * (1 - pf)     ).ravel('C').tolist()

    # Deposition
    deposition_fast_list = (-sign * dep_soc_fast).ravel('C').tolist()
    deposition_slow_list = (-sign * dep_soc_slow).ravel('C').tolist()

    # Vegetation）
    vegetation_fast_list = (-sign * V * V_FAST_PROP      ).ravel('C').tolist()
    vegetation_slow_list = (-sign * V * (1 - V_FAST_PROP)).ravel('C').tolist()

    # Reaction
    reaction_fast_list = (sign * INIT_VALUES.K_fast * MAP_STATS.C_fast_current ).ravel('C').tolist()
    reaction_slow_list = (sign * INIT_VALUES.K_slow * MAP_STATS.C_slow_current ).ravel('C').tolist()

    # RUSLE Factors
    E_t_ha_list   =  E_t_ha_month .ravel('C').tolist()
    C_factor_list =  C_factor_2D  .ravel('C').tolist()
    K_factor_list =  K_month      .ravel('C').tolist()
    LS_factor_list=  INIT_VALUES.LS_FACTOR    .ravel('C').tolist()
    P_factor_list =  INIT_VALUES.P_FACTOR     .ravel('C').tolist()
    R_factor_list =  R_month      .ravel('C').tolist()

    # Dams
    full_dams_list = MAP_STATS.full_dams    .ravel('C').tolist()
    dam_rem_cap_list = MAP_STATS.dam_rem_cap.ravel('C').tolist()
    dam_cur_stored_list = MAP_STATS.dam_cur_stored.ravel('C').tolist()
    
    df_out = pd.DataFrame({
        'LAT': lat_list,
        'LON': lon_list,
        'Landuse': landuse_list,
        'Region': Region_list,
        'Low point': Low_point_list,
        'C_fast': C_fast_list,
        'C_slow': C_slow_list,
        'Total_C': C_total_list,
        'Erosion_fast': erosion_fast_list,
        'Erosion_slow': erosion_slow_list,
        'Deposition_fast': deposition_fast_list,
        'Deposition_slow': deposition_slow_list,
        'Vegetation_fast': vegetation_fast_list,
        'Vegetation_slow': vegetation_slow_list,
        'Reaction_fast': reaction_fast_list,
        'Reaction_slow': reaction_slow_list,
        'E_t_ha_month': E_t_ha_list,
        'C_factor_month': C_factor_list,
        'K_factor_month': K_factor_list,
        'LS_factor_month': LS_factor_list,
        'P_factor_month': P_factor_list,
        'R_factor_month': R_factor_list,
        'Lost_SOC_River': lost_soc_list,
        'full_dam': full_dams_list,
        'dam_rem_cap': dam_rem_cap_list,
        'dam_cur_stored': dam_cur_stored_list
    })
    
    if USE_PARQUET:
        filename_parquet = f"SOC_terms_{year}_{month_idx+1:02d}_River.parquet"
        df_out.to_parquet(os.path.join(Paths.OUTPUT_DIR, "Data", filename_parquet), index=False, engine="pyarrow")
        print(f"Saved parquet output for Year {year}, Month {month_idx+1} as {filename_parquet}")
    else:
        filename_csv = f"SOC_terms_{year}_{month_idx+1:02d}_River.csv"
        df_out.to_csv(os.path.join(Paths.OUTPUT_DIR, "Data", filename_csv), index=False, float_format="%.6f")
        print(f"Saved CSV output for Year {year}, Month {month_idx+1} as {filename_csv}")
            