import sys
import os
import netCDF4 as nc
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import * 

USE_PARQUET = True

def compute_annual_sum_from_month(csv_folder, year, factor):
    total_A = 0.0
    column_name = factor + "_month"
    # 假设文件命名: SOC_terms_2007_{month:02d}_timestep_{month}_River.csv
    for month in range(1, 13):
        month_str = f"{month:02d}"
        # filename = rf"SOC_terms_{year}_{month_str}_timestep_(\d+)_River\.csv"
        if USE_PARQUET:
            filename = rf"SOC_terms_{year}_{month_str}_River\.parquet"
        else:
            filename = rf"SOC_terms_{year}_{month_str}_River\.csv"
        file_list = os.listdir(csv_folder)
        for f in file_list:
            if re.match(filename, f):
                # 读取 CSV
                if USE_PARQUET:
                    df_model = pd.read_parquet(os.path.join(csv_folder, f))
                else:
                    df_model = pd.read_csv(os.path.join(csv_folder, f))
                
                # 对当月的 factor 做汇总
                month_erosion_mean = df_model[column_name].mean()
                total_A += month_erosion_mean
                # print(f"monthly mean: {month_erosion_mean}")

    return total_A


start_year = 1992
end_year = 2018

plot_R = False
tongguan_tp = False

tp_list = []
R_list = []

csv_folder = OUTPUT_DIR / "Data"
tg_basin = gpd.read_file(DATA_DIR / "潼关以上流域.shp").to_crs("EPSG:4326")
minx, miny, maxx, maxy = tg_basin.total_bounds
# print(f"潼关经度范围: {minx:.4f}°E ~ {maxx:.4f}°E")
# print(f"潼关纬度范围: {miny:.4f}°N ~ {maxy:.4f}°N")


for year in range(start_year, end_year+1):
    nc_file = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{year}.nc"
    
    with nc.Dataset(nc_file) as ds:
        print(f"Processing year {year}...")
        if tongguan_tp:
            lon = ds.variables['longitude'][:]  # 1D array
            lat = ds.variables['latitude'][:]  # 1D array
            tp_data = ds.variables['tp'][:]  # shape (12, n_points)，单位：米
            tp_data = tp_data * 30           # 乘以30天，估算月降雨
            tp_data_mm = tp_data * 1000.0    # 转成毫米
            
            total_tp = 0.0
            
            # 每月分别clip后计算
            for month_idx in range(12):
                tp_month = tp_data_mm[month_idx, :]  # (n_points,)
                
                # 创建对应点的GeoDataFrame
                gdf = gpd.GeoDataFrame({
                    'lon': lon,
                    'lat': lat,
                    'tp': tp_month
                }, geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
                
                # 只保留落在潼关流域内的点
                gdf_clip = gpd.sjoin(gdf, tg_basin, predicate='within', how='inner')
                
                if not gdf_clip.empty:
                    month_mean_tp = gdf_clip['tp'].mean()  # 流域内点的平均降雨量
                    total_tp += month_mean_tp
                    lon_min, lat_min = gdf_clip['lon'].min(), gdf_clip['lat'].min()
                    lon_max, lat_max = gdf_clip['lon'].max(), gdf_clip['lat'].max()
                    # print(f"裁剪后点的经纬度范围: [{lon_min}, {lat_min}], [{lon_max}, {lat_max}]")
                else:
                    print(f"No data inside basin for {year} month {month_idx+1}")
                
                # fig, ax = plt.subplots(figsize=(8, 6))
                # tg_basin.boundary.plot(ax=ax, color='black', linewidth=1)  # 潼关流域边界
                # gdf.plot(ax=ax, color='blue', markersize=5, alpha=0.3, label='原始点')
                # gdf_clip.plot(ax=ax, color='red', markersize=5, label='裁剪后点')

                # plt.legend()
                # plt.title(f"{year}年{month_idx+1}月 降雨点位置")
                # plt.show()

            tp_list.append(total_tp)
            print(f"tp for year {year}: {total_tp:.2f}mm")
        
        else:
            tp_data = ds.variables['tp'][:]       # shape: (12, n_points), in meters
            tp_data = tp_data * 30
            tp_data_mm = tp_data * 1000.0
            tp_year = np.sum(tp_data_mm, axis=0)
            tp_year_mean = np.mean(tp_year)
            tp_list.append(tp_year_mean)
            print(f"tp for year {year}: {tp_year_mean:.2f}")
            
        if plot_R:
            R_mean = compute_annual_sum_from_month(csv_folder, year, "R_factor")
            R_list.append(R_mean)
            print(f"R for year {year}: {R_mean:.2f}")
        
plt.figure(figsize=(8, 5))
plt.plot(range(start_year, end_year+1), tp_list, marker='o', label='Annual Total Precipitation', linestyle='-')
if plot_R:
    plt.plot(range(start_year, end_year+1), R_list, marker='o', label='Annual Mean R Factor', linestyle='--')
plt.xlabel('Year')
plt.ylabel('tp(mm)')
plt.title('Annual Total Precipitation')
plt.legend()
plt.tight_layout()
if tongguan_tp:
    file_name = f"tongguan_tp_{start_year}-{end_year}.png"
else:
    file_name = f"tp_{start_year}-{end_year}.png"
plt.savefig(os.path.join(OUTPUT_DIR, file_name))
print(f"TP plot saved as: \"{file_name}\"")
plt.show()