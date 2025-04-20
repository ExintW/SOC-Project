import sys
import os
import netCDF4 as nc
import pandas as pd
import pandas.testing as pdt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
import pyarrow
from contextlib import nullcontext

from globalss import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import * 
from utils import *
from RUSLE_Calculations import *

from soil_and_soc_flow import distribute_soil_and_soc_with_dams_numba
from SOC_dynamics import vegetation_input, soc_dynamic_model


def run_simulation_year(year, LS_factor, P_factor, sorted_indices, past=False, future=False):
    # Filter dams built on or before current year
    df_dam_active = MAP_STATS.df_dam[MAP_STATS.df_dam["year"] <= year].copy()
    dam_capacity_arr = np.zeros(INIT_VALUES.DEM.shape, dtype=np.float64)
    for _, row in df_dam_active.iterrows():
        i_idx = find_nearest_index(MAP_STATS.grid_y, row["y"])
        j_idx = find_nearest_index(MAP_STATS.grid_x, row["x"])
        capacity_10000_m3 = row["capacity_remained"]
        capacity_tons = capacity_10000_m3 * 10000 * BULK_DENSITY
        dam_capacity_arr[i_idx, j_idx] = capacity_tons

    # Load monthly climate data (NetCDF)
    if future:
        lai_file = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_2015-2100_126.nc"
        nc_file  = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_pr_2015-2100_126.nc"
    else:
        nc_file = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{year}.nc"
    
    if not os.path.exists(nc_file):
        print(f"NetCDF file not found for year {year}: {nc_file}")
        return

    with nc.Dataset(nc_file) as ds, (nc.Dataset(lai_file) if future else nullcontext()) as ds_pr:
        # valid_time = ds.variables['valid_time'][:]  # Expect 12 months
        # n_time = len(valid_time)
        n_time = 12
        
        if future:
            # LAI file variables
            lon_nc = ds.variables['lon'][:]  # Adjusted variable name if needed
            lat_nc = ds.variables['lat'][:]
            lai_data = ds.variables['lai'][:]      # shape: (time, n_points)

            # Precipitation file variables
            lon_nc_pr = ds_pr.variables['lon'][:]
            lat_nc_pr = ds_pr.variables['lat'][:]
            pr_data = ds_pr.variables['pr'][:]         # shape: (time, n_points), in kg m^-2 s^-1
            tp_data_mm = pr_data * 30 * 86400     
            R_annual = calculate_r_factor_annually(tp_data_mm)
            R_annual_temp = create_grid_from_points(lon_nc_pr, lat_nc_pr, R_annual, MAP_STATS.grid_x, MAP_STATS.grid_y)
        
        else:
            lon_nc = ds.variables['longitude'][:]
            lat_nc = ds.variables['latitude'][:]
            lai_data = ds.variables['lai_lv'][:]  # shape: (12, n_points)
            tp_data = ds.variables['tp'][:]       # shape: (12, n_points), in meters
            tp_data = tp_data * 30
            tp_data_mm = tp_data * 1000.0
            R_annual = calculate_r_factor_annually(tp_data_mm)
            R_annual_temp = create_grid_from_points(lon_nc, lat_nc, R_annual, MAP_STATS.grid_x, MAP_STATS.grid_y)
        
        R_annual_temp = np.nan_to_num(R_annual_temp, nan=np.nanmean(R_annual_temp))
        print(f"Total elements in R Year: {R_annual_temp.size}, with max = {np.max(R_annual_temp)}, min = {np.min(R_annual_temp)}, mean = {np.mean(R_annual_temp)}")
    
        E_month_avg_list = []   # for calculating annual mean for validation
        
        if past:
            time_range = range(n_time-1, -1, -1)
        else:
            time_range = range(n_time)
            
        for month_idx in time_range:
            # Regrid LAI data
            time_month = time.time()
            
            lai_1d = lai_data[month_idx, :]
            LAI_2D = create_grid_from_points(lon_nc, lat_nc, lai_1d, MAP_STATS.grid_x, MAP_STATS.grid_y)
            LAI_2D = np.nan_to_num(LAI_2D, nan=np.nanmean(LAI_2D))
            
            print(f"\nLAI_1d regrid took {time.time() - time_month} seconds")
            time1 = time.time()
            
            # Regrid precipitation and convert to mm
            tp_1d_mm = tp_data_mm[month_idx, :]
            if future:
                RAIN_2D = create_grid_from_points(lon_nc_pr, lat_nc_pr, tp_1d_mm, MAP_STATS.grid_x, MAP_STATS.grid_y)
            else:
                RAIN_2D = create_grid_from_points(lon_nc, lat_nc, tp_1d_mm, MAP_STATS.grid_x, MAP_STATS.grid_y)
            RAIN_2D = np.nan_to_num(RAIN_2D, nan=np.nanmean(RAIN_2D))
            
            time2 = time.time()
            print(f"tp_1d regrid took {time2 - time1} seconds")

            # Compute RUSLE factors
            # R_month = calculate_r_factor_monthly(RAIN_2D)
            R_month = get_montly_r_factor(R_annual, tp_1d_mm, tp_data_mm)
            R_month = create_grid_from_points(lon_nc, lat_nc, R_month, MAP_STATS.grid_x, MAP_STATS.grid_y)
            R_month = np.nan_to_num(R_month, nan=np.nanmean(R_month))
            
            print(f"R_month regrid took {time.time() - time2} seconds")
            print(f"before R_month took {time.time() - time_month} seconds\n")

            print(f"Total elements in R month: {R_month.size}, with max = {np.max(R_month)}, min = {np.min(R_month)}, and mean = {np.mean(R_month)}")
            
            C_factor_2D = calculate_c_factor(LAI_2D)
            print(f"Total elements in C: {C_factor_2D.size}, with max = {np.max(C_factor_2D)}, min = {np.min(C_factor_2D)}, and mean = {np.mean(C_factor_2D)}")
            
            # Calculate monthly K factor
            K_month = calculate_k_factor(INIT_VALUES.SILT, INIT_VALUES.SAND, INIT_VALUES.CLAY, (MAP_STATS.C_fast_current + MAP_STATS.C_slow_current), INIT_VALUES.LANDUSE)
            K_month = np.nan_to_num(K_month, nan=np.nanmean(K_month))
            K_month = np.clip(K_month, 0, 0.7)
            print(f"Total elements in K: {K_month.size}, with max = {np.max(K_month)}, min = {np.min(K_month)}, and mean = {np.mean(K_month)}")

            # Calculate soil loss (t/ha/month) & then per cell
            E_t_ha_month = R_month * K_month * LS_factor * C_factor_2D * P_factor
            print(f"Total elements in E: {E_t_ha_month.size}, with max = {np.max(E_t_ha_month)}, min = {np.min(E_t_ha_month)}, and mean = {np.mean(E_t_ha_month)}")
            E_tcell_month = E_t_ha_month * CELL_AREA_HA
            E_month_avg_list.append(np.mean(E_t_ha_month))

            # Compute SOC mass eroded (kg/cell/month)
            S = E_tcell_month * (MAP_STATS.C_fast_current + MAP_STATS.C_slow_current)
            SOC_loss_g_kg_month = convert_soil_loss_to_soc_loss_monthly(
                E_t_ha_month, (MAP_STATS.C_fast_current + MAP_STATS.C_slow_current)
            )
            
            time1 = time.time()
            # Call the Numba-accelerated routing function
            D_soil, D_soc, inflow_soil, inflow_soc, lost_soc = distribute_soil_and_soc_with_dams_numba(
                E_tcell_month, S, INIT_VALUES.DEM, dam_capacity_arr, MAP_STATS.grid_x, MAP_STATS.grid_y,
                MAP_STATS.small_boundary_mask, MAP_STATS.small_outlet_mask,
                MAP_STATS.large_boundary_mask, MAP_STATS.large_outlet_mask,
                MAP_STATS.river_mask, sorted_indices,
                reverse=past
            )
            print(f"distribute soc took {time.time() - time1} seconds")

            # Debug: Print lost SOC summary
            mean_lost = np.mean(np.nan_to_num(lost_soc, nan=0))
            max_lost = np.nanmax(lost_soc)
            min_lost = np.nanmin(lost_soc)
            print(f"Year {year} Month {month_idx+1}: Lost_SOC - mean: {mean_lost:.2f}, "
                  f"max: {max_lost:.2f}, min: {min_lost:.2f}")

            # Compute vegetation input
            V = vegetation_input(LAI_2D)
            
            mean_gain = np.mean(np.nan_to_num(V, nan=0))
            max_gain = np.nanmax(V)
            min_gain = np.nanmin(V)
            print(f"Year {year} Month {month_idx+1}: SOC gain - mean: {mean_gain:.2f}, "
                  f"max: {max_gain:.2f}, min: {min_gain:.2f}")
            
            if past:
                dt = -1
            else:
                dt = 1
                
            # Update SOC pools
            MAP_STATS.C_fast_current, MAP_STATS.C_slow_current = soc_dynamic_model(
                MAP_STATS.C_fast_current, MAP_STATS.C_slow_current,
                SOC_loss_g_kg_month, D_soil, D_soc, V,
                INIT_VALUES.K_fast, INIT_VALUES.K_slow, MAP_STATS.p_fast_grid,
                dt=dt,
                M_soil=M_soil,
                lost_soc=lost_soc
            )

            # global_timestep += 1
            print(f"Completed simulation for Year {year}, Month {month_idx+1}")

            time1 = time.time()
            # Save figure output
            fig, ax = plt.subplots()
            cax = ax.imshow(MAP_STATS.C_fast_current + MAP_STATS.C_slow_current, cmap="viridis",
                            extent=[MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(), MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()],
                            origin='upper')
            cbar = fig.colorbar(cax, label="SOC (g/kg)")
            ax.set_title(f"SOC at Timestep Year {year}, Month {month_idx+1}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='x')
            filename_fig = f"SOC_{year}_{month_idx+1:02d}_River.png"
            plt.savefig(os.path.join(OUTPUT_DIR, "Figure", filename_fig))
            plt.close(fig)
            
            print(f"plot took {time.time() - time1} seconds")
            
            time1 = time.time()

            rows_grid, cols_grid = MAP_STATS.C_fast_current.shape
            lat_grid, lon_grid = np.meshgrid(MAP_STATS.grid_y, MAP_STATS.grid_x, indexing='ij')

            lat_list =  lat_grid.ravel(order='C').tolist()      # 与 for i→for j 顺序一致
            lon_list =  lon_grid.ravel(order='C').tolist()
            landuse_list = INIT_VALUES.LANDUSE.astype(str).ravel(order='C').tolist()

            pf = MAP_STATS.p_fast_grid
            sign = 1 if past else -1                              # past=True ➜ 正号，False ➜ 取反
            dep_conc = (D_soc * 1000.0) / M_soil                 # g kg‑1 → g kg‑1（与原式相同）

            # SOC 组分
            C_fast_list  = MAP_STATS.C_fast_current .ravel('C').tolist()
            C_slow_list  = MAP_STATS.C_slow_current .ravel('C').tolist()

            # Erosion（侵蚀输出）
            erosion_fast_list = ( sign * SOC_loss_g_kg_month *  pf          ).ravel('C').tolist()
            erosion_slow_list = ( sign * SOC_loss_g_kg_month * (1 - pf)     ).ravel('C').tolist()

            # Deposition（沉积输入，符号与 erosion 相反）
            deposition_fast_list = (-sign * dep_conc * pf          ).ravel('C').tolist()
            deposition_slow_list = (-sign * dep_conc * (1 - pf)    ).ravel('C').tolist()

            # Vegetation（植被输入）
            vegetation_fast_list = (V * pf         ).ravel('C').tolist()
            vegetation_slow_list = (V * (1 - pf)   ).ravel('C').tolist()

            # Reaction（微生物矿化）
            reaction_fast_list = (-INIT_VALUES.K_fast * MAP_STATS.C_fast_current ).ravel('C').tolist()
            reaction_slow_list = (-INIT_VALUES.K_slow * MAP_STATS.C_slow_current ).ravel('C').tolist()

            # 其余月度/因子量
            E_t_ha_list   =  E_t_ha_month .ravel('C').tolist()
            C_factor_list =  C_factor_2D  .ravel('C').tolist()
            K_factor_list =  K_month      .ravel('C').tolist()
            LS_factor_list=  LS_factor    .ravel('C').tolist()
            P_factor_list =  P_factor     .ravel('C').tolist()
            R_factor_list =  R_month      .ravel('C').tolist()
            lost_soc_list =  lost_soc     .ravel('C').tolist()

            print(f"Gather data for csv took {time.time() - time1} seconds")
            
            df_out = pd.DataFrame({
                'LAT': lat_list,
                'LON': lon_list,
                'Landuse': landuse_list,
                'C_fast': C_fast_list,
                'C_slow': C_slow_list,
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
                'Lost_SOC_River': lost_soc_list
            })
            
            if USE_PARQUET:
                filename_parquet = f"SOC_terms_{year}_{month_idx+1:02d}_River.parquet"
                df_out.to_parquet(os.path.join(OUTPUT_DIR, "Data", filename_parquet), index=False, engine="pyarrow")
                print(f"Saved parquet output for Year {year}, Month {month_idx+1} as {filename_parquet}")
            else:
                filename_csv = f"SOC_terms_{year}_{month_idx+1:02d}_River.csv"
                df_out.to_csv(os.path.join(OUTPUT_DIR, "Data", filename_csv), index=False, float_format="%.6f")
                print(f"Saved CSV output for Year {year}, Month {month_idx+1} as {filename_csv}")
            
            print(f"This month took {time.time() - time_month} seconds")
            
        print(f"\nAnnual mean of E = {np.mean(E_month_avg_list)}\n")