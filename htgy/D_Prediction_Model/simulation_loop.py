import sys
import os
import netCDF4 as nc
import pandas as pd
import pandas.testing as pdt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
from shapely.geometry import LineString, MultiLineString
import pyarrow
from contextlib import nullcontext
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from globalss import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import * 
from utils import *
from RUSLE_Calculations import *

from soil_and_soc_flow import distribute_soil_and_soc_with_dams_numba
from SOC_dynamics import vegetation_input, soc_dynamic_model


def run_simulation_year(year, LS_factor, P_factor, sorted_indices, past=False, future=False, a=-1.7, b=1.61, c=1):
    # Filter dams built on or before current year
    df_dam_active = MAP_STATS.df_dam[MAP_STATS.df_dam["year"] <= year].copy()
    active_dams = np.zeros(INIT_VALUES.DEM.shape, dtype=int)
    dam_max_cap = np.zeros(INIT_VALUES.DEM.shape, dtype=np.float64)
    dam_cur_stored = MAP_STATS.dam_cur_stored
    dam_capacity_arr = np.zeros(INIT_VALUES.DEM.shape, dtype=np.float64)
    
    for _, row in df_dam_active.iterrows():
        i_idx = find_nearest_index(MAP_STATS.grid_y, row["y"])
        j_idx = find_nearest_index(MAP_STATS.grid_x, row["x"])
        capacity_10000_m3 = row["capacity_remained"]
        capacity_tons = capacity_10000_m3 * 10000 * BULK_DENSITY
        dam_capacity_arr[i_idx, j_idx] = capacity_tons
        
        max_cap_10000_m3 = row["total_stor"]
        max_cap_tons = max_cap_10000_m3 * 10000 * BULK_DENSITY
        dam_max_cap[i_idx, j_idx] = max_cap_tons
        
        if dam_cur_stored[i_idx, j_idx] == 0.0:   # Only initialize cur_stored for new dams for this year
            cur_stored_10000_m3 = row["deposition"]
            cur_stored_tons = cur_stored_10000_m3 * 10000 * BULK_DENSITY
            if np.isnan(cur_stored_tons):
                cur_stored_tons = 0.0
            dam_cur_stored[i_idx, j_idx] = cur_stored_tons
        
        active_dams[i_idx, j_idx] = 1

    # Load monthly climate data (NetCDF)
    if future:
        pr_file  = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_pr_points_2015-2100_126.nc"
    else:
        nc_file = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{year}.nc"

    if year <= 2000:
        lai_file = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_1950-2000.nc"
        cmip_start = 1950
    elif year <= 2014:
        lai_file = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2001-2014.nc"
        cmip_start = 2001
    else:
        lai_file = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_126.nc"
        cmip_start = 2015

    if future != True:
        if not os.path.exists(nc_file):
            print(f"NetCDF file not found for year {year}: {nc_file}")
            return

    with (nc.Dataset(nc_file) if not future else nullcontext()) as ds, nc.Dataset(lai_file) as ds_lai, (nc.Dataset(pr_file) if future else nullcontext()) as ds_pr:

        # valid_time = ds.variables['valid_time'][:]  # Expect 12 months
        # n_time = len(valid_time)
        n_time = 12

        if future:
            # LAI file variables

            lon_lai = ds_lai.variables['lon'][:]  # Adjusted variable name if needed
            lat_lai = ds_lai.variables['lat'][:]
            lai_data = ds_lai.variables['lai'][:]  # shape: (12, n_points)

            lon_nc = lon_lai
            lat_nc = lat_lai

            # Precipitation file variables
            lon_nc_pr = ds_pr.variables['lon'][:]
            lat_nc_pr = ds_pr.variables['lat'][:]
            pr_data = ds_pr.variables['pr'][:]         # shape: (time, n_points), in kg m^-2 s^-1
            tp_data_mm = pr_data * 30 * 86400

            # figure out which 12‐month block corresponds to `year`
            start_idx = (year - cmip_start) * n_time
            end_idx = start_idx + n_time

            if start_idx < 0 or end_idx > tp_data_mm.shape[0]:
                raise ValueError(f"No CMIP6 data for year {year} (idx {start_idx}:{end_idx})")

                # slice out the 12 months, no summing
            annual_tp_data_mm = tp_data_mm[start_idx:end_idx, :]  # shape = (12, n_points)

            R_annual = calculate_r_factor_annually(annual_tp_data_mm, c=c, b=b)
            R_annual_temp = create_grid_from_points(lon_nc_pr, lat_nc_pr, R_annual, MAP_STATS.grid_x, MAP_STATS.grid_y)

        else:
            if USE_CMIP6:
                lon_lai = ds_lai.variables['lon'][:]  # Adjusted variable name if needed
                lat_lai = ds_lai.variables['lat'][:]
                lai_data = ds_lai.variables['lai'][:]  # shape: (12, n_points)
            else:
                lon_lai = ds.variables['longitude'][:]  # Adjusted variable name if needed
                lat_lai = ds.variables['latitude'][:]
                lai_data = ds.variables['lai_lv'][:]  # shape: (12, n_points)

            lon_nc = ds.variables['longitude'][:]
            lat_nc = ds.variables['latitude'][:]
            tp_data = ds.variables['tp'][:]       # shape: (12, n_points), in meters
            tp_data = tp_data * 30
            tp_data_mm = tp_data * 1000.0
            R_annual = calculate_r_factor_annually(tp_data_mm, c=c, b=b)
            R_annual_temp = create_grid_from_points(lon_nc, lat_nc, R_annual, MAP_STATS.grid_x, MAP_STATS.grid_y)

        R_annual_temp = np.nan_to_num(R_annual_temp, nan=np.nanmean(R_annual_temp))
        R_annual_temp[~MAP_STATS.loess_border_mask] = np.nan
        print(f"Total elements in R Year: {R_annual_temp.size}, with max = {np.nanmax(R_annual_temp)}, min = {np.nanmin(R_annual_temp)}, mean = {np.nanmean(R_annual_temp)}")

        E_month_avg_list = []   # for calculating annual mean for validation
        C_month_list = []
        if past:
            time_range = range(n_time-1, -1, -1)    # 11 -> 0
        else:
            time_range = range(n_time)
            
        if USE_UNET and past:
            UNet_Model = INIT_VALUES.UNet_Model

        for month_idx in time_range:
            idx = month_idx

            # CMIP6 / future case: compute flat index relative to cmip_start
            if USE_CMIP6 or future:
                idx = (year - cmip_start) * n_time + month_idx
                # skip months before or after our CMIP6 file
                if idx < 0 or idx >= lai_data.shape[0]:
                    continue
            # Regrid LAI data
            time_month = time.time()

            print(f"\n=======================================================================")
            print(f"                          Year {year} Month {month_idx+1}")
            print(f"=======================================================================\n")

            print(f"lai_data shape = {lai_data.shape}")
            lai_1d = lai_data[idx, :]
            LAI_2D = create_grid_from_points(lon_lai, lat_lai, lai_1d, MAP_STATS.grid_x, MAP_STATS.grid_y)
            LAI_2D = np.nan_to_num(LAI_2D, nan=np.nanmean(LAI_2D))
            
            print(f"LAI_1d regrid took {time.time() - time_month} seconds")
            time1 = time.time()
            
            # Regrid precipitation and convert to mm
            if future:
                tp_1d_mm = tp_data_mm[idx, :]
                RAIN_2D = create_grid_from_points(lon_nc_pr, lat_nc_pr, tp_1d_mm, MAP_STATS.grid_x, MAP_STATS.grid_y)
            else:
                tp_1d_mm = tp_data_mm[month_idx, :]
                RAIN_2D = create_grid_from_points(lon_nc, lat_nc, tp_1d_mm, MAP_STATS.grid_x, MAP_STATS.grid_y)
            RAIN_2D = np.nan_to_num(RAIN_2D, nan=np.nanmean(RAIN_2D))
            
            time2 = time.time()
            print(f"tp_1d regrid took {time2 - time1} seconds")

            # Compute RUSLE factors
            # R_month = calculate_r_factor_monthly(RAIN_2D)
            if future:
                R_month = get_montly_r_factor(R_annual, tp_1d_mm, annual_tp_data_mm)
            else:
                R_month = get_montly_r_factor(R_annual, tp_1d_mm, tp_data_mm)
            R_month = create_grid_from_points(lon_nc, lat_nc, R_month, MAP_STATS.grid_x, MAP_STATS.grid_y)
            R_month = np.nan_to_num(R_month, nan=np.nanmean(R_month))
            R_month[~MAP_STATS.loess_border_mask] = np.nan
            
            print(f"R_month regrid took {time.time() - time2} seconds")
            print(f"before R_month took {time.time() - time_month} seconds\n")

            print(f"Total elements in R month: {R_month.size}, with max = {np.nanmax(R_month)}, min = {np.nanmin(R_month)}, and mean = {np.nanmean(R_month)}")
            
            C_factor_2D = calculate_c_factor(LAI_2D, a=a)
            C_factor_2D[~MAP_STATS.loess_border_mask] = np.nan
            C_month_list.append(np.nanmean(C_factor_2D))
            print(f"Total elements in C: {C_factor_2D.size}, with max = {np.nanmax(C_factor_2D)}, min = {np.nanmin(C_factor_2D)}, and mean = {np.nanmean(C_factor_2D)}")
            print(f"\nLAI mean = {np.nanmean(LAI_2D)}\n")
            
            # Calculate monthly K factor
            K_month = calculate_k_factor(INIT_VALUES.SILT, INIT_VALUES.SAND, INIT_VALUES.CLAY, (MAP_STATS.C_fast_current + MAP_STATS.C_slow_current), INIT_VALUES.LANDUSE)
            K_month = np.nan_to_num(K_month, nan=np.nanmean(K_month))
            K_month[~MAP_STATS.loess_border_mask] = np.nan
            print(f"Total elements in K: {K_month.size}, with max = {np.nanmax(K_month)}, min = {np.nanmin(K_month)}, and mean = {np.nanmean(K_month)}")

            # Calculate soil loss (t/ha/month) & then per cell
            E_t_ha_month = R_month * K_month * LS_factor * C_factor_2D * P_factor
            E_t_ha_month[~MAP_STATS.loess_border_mask] = np.nan
            print(f"Total elements in E: {E_t_ha_month.size}, with max = {np.nanmax(E_t_ha_month)}, min = {np.nanmin(E_t_ha_month)}, and mean = {np.nanmean(E_t_ha_month)}")
            E_tcell_month = E_t_ha_month * CELL_AREA_HA
            E_month_avg_list.append(np.nanmean(E_t_ha_month))

            # Compute SOC mass eroded (kg/cell/month)
            S = E_tcell_month * (MAP_STATS.C_fast_current + MAP_STATS.C_slow_current)
            SOC_loss_g_kg_month = convert_soil_loss_to_soc_loss_monthly(
                E_t_ha_month, (MAP_STATS.C_fast_current + MAP_STATS.C_slow_current)
            )
            
            A = convert_soil_to_soc_loss(E_t_ha_month)
            
            # # Call the Numba-accelerated routing function
            # D_soil, D_soc, inflow_soil, inflow_soc, lost_soc = distribute_soil_and_soc_with_dams_numba(
            #     E_tcell_month, S, INIT_VALUES.DEM, dam_capacity_arr, MAP_STATS.grid_x, MAP_STATS.grid_y,
            #     MAP_STATS.small_boundary_mask, MAP_STATS.small_outlet_mask,
            #     MAP_STATS.large_boundary_mask, MAP_STATS.large_outlet_mask,
            #     MAP_STATS.river_mask, sorted_indices,
            #     reverse=past
            # )

            # # Debug: Print SOC summary
            # Lost_soc_concentration = lost_soc*1000/M_soil
            # mean_river_lost = np.nanmean(np.nan_to_num(Lost_soc_concentration, nan=0))
            # max_river_lost = np.nanmax(Lost_soc_concentration)
            # min_river_lost = np.nanmin(Lost_soc_concentration)
            # print(f"Year {year} Month {month_idx+1}: River_Lost_SOC - mean: {mean_river_lost:.2f}, "
            #     f"max: {max_river_lost:.2f}, min: {min_river_lost:.2f}")

            # mean_erosion_lost = np.nanmean(np.nan_to_num(SOC_loss_g_kg_month, nan=0))
            # max_erosion_lost = np.nanmax(SOC_loss_g_kg_month)
            # min_erosion_lost = np.nanmin(SOC_loss_g_kg_month)
            # print(f"Year {year} Month {month_idx + 1}: Erosion_Lost_SOC - mean: {mean_erosion_lost:.2f}, "
            #     f"max: {max_erosion_lost:.2f}, min: {min_erosion_lost:.2f}")

            # Compute vegetation input
            V = vegetation_input(LAI_2D)
            
            mean_vege_gain = np.nanmean(np.nan_to_num(V, nan=0))
            # max_vege_gain = np.nanmax(V)
            # min_vege_gain = np.nanmin(V)
            print(f"Year {year} Month {month_idx+1}: SOC_Vegetation_Gain - mean: {mean_vege_gain:.2f}, ")
            #       f"max: {max_vege_gain:.2f}, min: {min_vege_gain:.2f}")

            # deposition_SOC_gain = D_soc*1000/M_soil
            # mean_deposition_gain = np.nanmean(np.nan_to_num(deposition_SOC_gain, nan=0))
            # max_deposition_gain = np.nanmax(deposition_SOC_gain)
            # min_deposition_gain = np.nanmin(deposition_SOC_gain)
            # print(f"Year {year} Month {month_idx + 1}: SOC_deposition_Gain - mean: {mean_deposition_gain:.2f}, "
            #       f"max: {max_deposition_gain:.2f}, min: {min_deposition_gain:.2f}")

            # mean_K_fast = np.nanmean(np.nan_to_num(INIT_VALUES.K_fast, nan=0))
            # max_K_fast = np.nanmax(INIT_VALUES.K_fast)
            # min_K_fast = np.nanmin(INIT_VALUES.K_fast)
            # print(f"Year {year} Month {month_idx + 1}: K_fast mean: {mean_K_fast:.6f}, "
            #       f"max: {max_K_fast:.6f}, min: {min_K_fast:.6f}")

            # mean_K_slow = np.nanmean(np.nan_to_num(INIT_VALUES.K_slow, nan=0))
            # max_K_slow = np.nanmax(INIT_VALUES.K_slow)
            # min_K_slow = np.nanmin(INIT_VALUES.K_slow)
            # print(f"Year {year} Month {month_idx + 1}: K_fast mean: {mean_K_slow:.6f}, "
            #       f"max: {max_K_slow:.6f}, min: {min_K_slow:.6f}")

            # reaction_fast_loss = INIT_VALUES.K_fast * MAP_STATS.C_fast_current
            # mean_reaction_fast_loss = np.nanmean(np.nan_to_num(reaction_fast_loss, nan=0))
            # max_reaction_fast_loss = np.nanmax(reaction_fast_loss)
            # min_reaction_fast_loss = np.nanmin(reaction_fast_loss)
            # print(f"Year {year} Month {month_idx + 1}: SOC_Reaction_Fast_Loss - mean: {mean_reaction_fast_loss:.2f}, "
            #       f"max: {max_reaction_fast_loss:.2f}, min: {min_reaction_fast_loss:.2f}")

            # reaction_slow_loss = INIT_VALUES.K_slow * MAP_STATS.C_slow_current
            # mean_reaction_slow_loss = np.nanmean(np.nan_to_num(reaction_slow_loss, nan=0))
            # max_reaction_slow_loss = np.nanmax(reaction_slow_loss)
            # min_reaction_slow_loss = np.nanmin(reaction_slow_loss)
            # print(f"Year {year} Month {month_idx + 1}: SOC_Reaction_Slow_Loss - mean: {mean_reaction_slow_loss:.2f}, "
            #       f"max: {max_reaction_slow_loss:.2f}, min: {min_reaction_slow_loss:.2f}")

            # print(f"Year {year} Month {month_idx + 1}: SOC_mean_change: {(mean_deposition_gain + mean_vege_gain - mean_reaction_fast_loss - mean_reaction_slow_loss - mean_erosion_lost - mean_river_lost):.2f} ")

            soc_time = time.time()
            if past and USE_UNET:
                MAP_STATS.C_fast_current, MAP_STATS.C_slow_current, dep_soc_fast, dep_soc_slow, lost_soc = soc_dynamic_model(E_tcell_month, A, sorted_indices, dam_max_cap, dam_cur_stored, active_dams, V, month_idx, year, past, UNet_MODEL=UNet_Model)
            elif past and USE_1980_LAI_TREND:
                LAI_2D[~MAP_STATS.loess_border_mask] = np.nan
                LAI_avg = np.nanmean(LAI_2D)
                MAP_STATS.C_fast_current, MAP_STATS.C_slow_current, dep_soc_fast, dep_soc_slow, lost_soc = soc_dynamic_model(E_tcell_month, A, sorted_indices, dam_max_cap, dam_cur_stored, active_dams, V, month_idx, year, past, LAI_avg=LAI_avg)
            else:
                MAP_STATS.C_fast_current, MAP_STATS.C_slow_current, dep_soc_fast, dep_soc_slow, lost_soc = soc_dynamic_model(E_tcell_month, A, sorted_indices, dam_max_cap, dam_cur_stored, active_dams, V, month_idx, year, past)
            print(f'SOC took {time.time() - soc_time}')
            
            if year == EQUIL_YEAR and not past:
                MAP_STATS.C_fast_equil_list.append(MAP_STATS.C_fast_current)
                MAP_STATS.C_slow_equil_list.append(MAP_STATS.C_slow_current)
            
            # print(f"C fast nan: {np.isnan(MAP_STATS.C_fast_current).sum()}")
            # print(f"C slow nan: {np.isnan(MAP_STATS.C_slow_current).sum()}")
            
            C_total = MAP_STATS.C_fast_current + MAP_STATS.C_slow_current
            mean_C_total = np.nanmean(C_total)
            max_C_total = np.nanmax(C_total)
            min_C_total = np.nanmin(C_total)

            # stash a copy of this month’s total‐C grid
            if past:
                MAP_STATS.total_C_matrix.insert(0, C_total.copy())
                MAP_STATS.C_fast_matrix.insert(0, MAP_STATS.C_fast_current.copy())
                MAP_STATS.C_slow_matrix.insert(0, MAP_STATS.C_slow_current.copy())
                MAP_STATS.active_dam_matrix.insert(0, active_dams.copy())
            else:
                MAP_STATS.total_C_matrix.append(C_total.copy())
                MAP_STATS.C_fast_matrix.append(MAP_STATS.C_fast_current.copy())
                MAP_STATS.C_slow_matrix.append(MAP_STATS.C_slow_current.copy())
                MAP_STATS.active_dam_matrix.append(active_dams.copy())


            # New: count and report cells where C_total > 40 and it's an active dam
            high_dam_mask = (C_total > 40) & active_dams
            count_high_dam = np.count_nonzero(high_dam_mask)
            print(f"Number of active dam cells with C_total > 40: {count_high_dam}")

            print(f"Year {year} Month {month_idx + 1}: Total_SOC_mean: {mean_C_total:.2f}, "
                  f"max: {max_C_total:.2f}, min: {min_C_total:.2f}")

            # global_timestep += 1
            print(f"Completed simulation for Year {year}, Month {month_idx+1}")

            time1 = time.time()
            # Save figure output
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            # ─── FIGURE OUTPUT ─────────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(10, 6))

            # 1) Plot SOC
            im = ax.imshow(
                MAP_STATS.C_fast_current + MAP_STATS.C_slow_current,
                cmap="viridis", vmin=0, vmax=30,
                extent=[
                    MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(),
                    MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()
                ],
                origin="upper"
            )

            # 2) Overlay the border (no fill, just outline)
            border = MAP_STATS.loess_border_geom.boundary
            if isinstance(border, LineString):
                x, y = border.xy
                ax.plot(x, y, color="black", linewidth=0.4)
            elif isinstance(border, MultiLineString):
                for seg in border.geoms:
                    x, y = seg.xy
                    ax.plot(x, y, color="black", linewidth=0.4)

            # 3) Append a colorbar axis the same height as the map, with padding
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad="4%")
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label("SOC (g/kg)")

            # 4) Labels and formatting
            ax.set_title(f"SOC at Timestep Year {year}, Month {month_idx + 1}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style="plain", axis="x")

            # 5) Save and close
            filename_fig = f"SOC_{year}_{month_idx + 1:02d}_River.png"
            plt.savefig(
                os.path.join(OUTPUT_DIR, "Figure", filename_fig),
                dpi=600,
                bbox_inches="tight"
            )
            plt.close(fig)

            print(f"plot took {time.time() - time1} seconds")
            
            time1 = time.time()

            rows_grid, cols_grid = MAP_STATS.C_fast_current.shape
            lat_grid, lon_grid = np.meshgrid(MAP_STATS.grid_y, MAP_STATS.grid_x, indexing='ij')

            lat_list =  lat_grid.ravel(order='C').tolist()      # 与 for i→for j 顺序一致
            lon_list =  lon_grid.ravel(order='C').tolist()
            landuse_list = INIT_VALUES.LANDUSE.astype(str).ravel(order='C').tolist()
            Region_list = INIT_VALUES.REGION.astype(str).ravel(order='C').tolist()
            Low_point_list = MAP_STATS.low_mask.astype(str).ravel('C').tolist()

            pf = MAP_STATS.p_fast_grid
            sign = 1 if past else -1                              # past=True ➜ 正号，False ➜ 取反

            # SOC 组分
            C_fast_list  = MAP_STATS.C_fast_current .ravel('C').tolist()
            C_slow_list  = MAP_STATS.C_slow_current .ravel('C').tolist()

            # Erosion（侵蚀输出）
            erosion_fast_list = ( sign * SOC_loss_g_kg_month *  pf          ).ravel('C').tolist()
            erosion_slow_list = ( sign * SOC_loss_g_kg_month * (1 - pf)     ).ravel('C').tolist()

            # Deposition（沉积输入，符号与 erosion 相反）
            deposition_fast_list = (-sign * dep_soc_fast).ravel('C').tolist()
            deposition_slow_list = (-sign * dep_soc_slow).ravel('C').tolist()

            # Vegetation（植被输入）
            vegetation_fast_list = (V * V_FAST_PROP      ).ravel('C').tolist()
            vegetation_slow_list = (V * (1 - V_FAST_PROP)).ravel('C').tolist()

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

            C_total_list = C_total       .ravel('C').tolist()

            print(f"Gather data for csv took {time.time() - time1} seconds")
            
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
            
        print(f'C factor annual avg = {np.nanmean(C_month_list)}')
        print(f"\nAnnual mean of E = {np.nanmean(E_month_avg_list)}\n")