import sys
import os
import netCDF4 as nc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


start_year = 1980
end_year = 2018

plot_R = False

tp_list = []
R_list = []

csv_folder = OUTPUT_DIR / "Data"

for year in range(start_year, end_year+1):
    nc_file = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{year}.nc"
    
    with nc.Dataset(nc_file) as ds:
        print(f"Processing year {year}...")
        tp_data = ds.variables['tp'][:]       # shape: (12, n_points), in meters
        tp_data = tp_data * 30
        tp_data_mm = tp_data * 1000.0
        tp_year = np.sum(tp_data_mm, axis=0)
        tp_year_mean = np.mean(tp_year)
        tp_list.append(tp_year_mean)
        print(f"tp for year {year}: {tp_year_mean}")
        
        if plot_R:
            R_mean = compute_annual_sum_from_month(csv_folder, year, "R_factor")
            R_list.append(R_mean)
            print(f"R for year {year}: {R_mean}")
        
plt.figure(figsize=(8, 5))
plt.plot(range(start_year, end_year+1), tp_list, marker='o', label='Annual Total Precipitation', linestyle='-')
if plot_R:
    plt.plot(range(start_year, end_year+1), R_list, marker='o', label='Annual Mean R Factor', linestyle='--')
plt.xlabel('Year')
plt.ylabel('tp(mm)')
plt.title('Annual Total Precipitation')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"tp_{start_year}-{end_year}.png"))
print(f"TP plot saved as: \"tp_{start_year}-{end_year}.png\"")
plt.show()