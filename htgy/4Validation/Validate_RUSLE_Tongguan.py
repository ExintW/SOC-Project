import sys
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import re
import matplotlib.pyplot as plt
from shapely.geometry import Point

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # 假设这里面定义了 OUTPUT_DIR, DATA_DIR 等

USE_PARQUET = True

def compute_annual_sum_from_month(csv_folder, year):
    total_A = 0.0
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
                    df = pd.read_parquet(os.path.join(csv_folder, f))
                else:
                    df = pd.read_csv(os.path.join(csv_folder, f))
                df = df.dropna(subset=["LON", "LAT", "E_t_ha_month"])
                gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["LON"], df["LAT"])], crs="EPSG:4326")
                
                tg_basin = gpd.read_file(DATA_DIR / "潼关以上流域.shp").to_crs("EPSG:4326")
                gdf_clip = gpd.sjoin(gdf, tg_basin, predicate='within')
                model_mean_A = gdf_clip["E_t_ha_month"].mean()
                
                total_A += model_mean_A
                # print(f"monthly mean: {month_erosion_mean}")

    return total_A

def get_tongguan_valid_data(data_path, year):
    valid_df = pd.read_excel(data_path, sheet_name=0)  # 假设含“年份”、“年总输沙量_t”等字段
    valid_year = valid_df[valid_df["年份"] == year].iloc[0]
    valid_sediment = valid_year["泥沙(10^8 t)"] * 10e8 # t
    
    tg_basin = gpd.read_file(DATA_DIR / "潼关以上流域.shp")
    basin_area_km2 = tg_basin.to_crs("EPSG:32649").area.sum() / 1e6  # km²
    basin_area_ha = basin_area_km2 * 100  # ha
    valid_A = valid_sediment / basin_area_ha  # 单位统一为 t/ha

    return valid_A

if __name__ == "__main__":
    start_year = 1950
    end_year = 2018
    time_step = 1

    years_list = []
    A_valid_list = []  
    A_model_list = []
    # mae_list = []
    # mre_list = []
    # af_list = []
    
    preload = False
    preload_file = PROCESSED_DIR / "tongguan_valid.npy"
    if preload_file.exists():
        preload = True
        print("Loading precomputed Tongguan valid data...")
        A_valid_list = np.load(preload_file)
    
    csv_dir = OUTPUT_DIR / "Data"
    valid_path = DATA_DIR / "潼关径流泥沙.xlsx"

    idx = 0
    for year in range(start_year, end_year+1, time_step):
        years_list.append(year)
        
        model_mean_A = compute_annual_sum_from_month(csv_dir, year)
        A_model_list.append(model_mean_A)
        
        if not preload:
            valid_mean_A = get_tongguan_valid_data(valid_path, year)
            A_valid_list.append(valid_mean_A)

        print(f"{year}年潼关实测值: {A_valid_list[idx]:.2f} t/ha")
        print(f"{year}年模型模拟值: {model_mean_A:.2f} t/ha")
        idx += 1
    
    if not preload:
        np.save(preload_file, A_valid_list)
    
    # Convert lists to numpy arrays
    A_model_array = np.array(A_model_list)
    A_valid_array = np.array(A_valid_list)

    # Calculate errors
    errors = A_model_array - A_valid_array
    # RMSE
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    # R²
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((A_valid_array - np.mean(A_valid_array))**2)
    r2 = 1 - (ss_res / ss_tot)
    # NSE
    nse = 1 - (np.sum(errors**2) / np.sum((A_valid_array - np.mean(A_valid_array))**2))
    # MAE
    mae = np.mean(np.abs(errors))
    # AF
    af = np.max([np.max(A_model_array / A_valid_array), np.max(A_valid_array / A_model_array)])

    # Output
    print("\n------总评估指标------")
    print(f"Total RMSE: {rmse:.4f} t/ha")
    print(f"Total R²: {r2:.4f}")
    print(f"Total NSE: {nse:.4f}")
    print(f"Total MAE: {mae:.4f} t/ha")
    print(f"Total AF: {af:.4f}\n")
        
    # === Plot 1: 模拟值 vs 实测值 ===
    plt.figure(figsize=(8, 5))
    plt.plot(years_list, A_model_list, marker='o', label='Modelled A')
    plt.plot(years_list, A_valid_list, marker='s', label='Tongguan A')
    plt.xlabel('Year')
    plt.ylabel('Soil Erosion Modulus (t/ha)')
    plt.title('Modelled vs Tongguan A (RUSLE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Tongguan_A_model_vs_valid.png")
    plt.show()
    plt.close()

    # # === Plot 2: 各种误差指标 ===
    # plt.figure(figsize=(10, 6))
    # plt.plot(years_list, mae_list, label='MAE', marker='o')
    # plt.plot(years_list, mre_list, label='MRE', marker='^')
    # plt.plot(years_list, af_list, label='Af', marker='d')
    # plt.xlabel('Year')
    # plt.ylabel('Error Metrics')
    # plt.title('Model Validation Error Metrics per Year')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(OUTPUT_DIR / "Tongguan_A_error_metrics.png")
    # plt.show()
    # plt.close()
