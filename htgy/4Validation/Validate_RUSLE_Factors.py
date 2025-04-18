import sys
import os
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import re
import math
import matplotlib.pyplot as plt  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # 假设这里面定义了 OUTPUT_DIR, DATA_DIR 等

# Validation Data Source: https://doi.org/10.57760/sciencedb.07135

def compute_mean_value(raster_path, shp_path):
    """
    读取 R2008_1000.tif，并使用 Loess_Plateau_vector_border.shp 掩膜，
    返回掩膜区域（黄土高原）有效像元的 R 值平均值。
    """
    # 读取矢量边界
    loess_gdf = gpd.read_file(shp_path)
    
    # 打开栅格文件，检查 CRS，如果不一致则需重投影矢量
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if loess_gdf.crs != raster_crs:
            loess_gdf = loess_gdf.to_crs(raster_crs)
        
        # 将矢量边界转为可用于 rasterio.mask.mask 的几何对象
        geoms = [feature["geometry"] for feature in loess_gdf.__geo_interface__["features"]]
        
        # 对栅格进行掩膜
        out_image, out_transform = rasterio.mask.mask(src, geoms, crop=True)
        # 这里 out_image 可能是一个三维数组，形如 (bands, height, width)
        # 如果只有一个波段，则 out_image.shape[0] == 1
        
        # 获取nodata值（若无定义，可自行设置掩膜阈值）
        nodata_val = src.nodata if src.nodata is not None else -9999
        

    # 计算掩膜后区域像元 R 的平均值
    # out_image[0, :, :] 为单波段，如果是单波段的栅格，index=0 取第0层
    masked_array = out_image[0]
    
    # 若nodata没有定义，可能还需要基于掩膜区域将无效像元移除
    # 这里示例假设nodata_val为nodata或背景值
    valid_mask = (masked_array != nodata_val) & (~np.isnan(masked_array))
    mean_r = masked_array[valid_mask].mean()


    return mean_r

def compute_annual_mean_from_month(csv_folder, year, factor):
    total_A = []
    column_name = factor + "_month"
    # 假设文件命名: SOC_terms_2007_{month:02d}_timestep_{month}_River.csv
    for month in range(1, 13):
        month_str = f"{month:02d}"
        filename = rf"SOC_terms_{year}_{month_str}_timestep_(\d+)_River\.csv"
        file_list = os.listdir(csv_folder)
        for f in file_list:
            if re.match(filename, f):
                # 读取 CSV
                df_model = pd.read_csv(os.path.join(csv_folder, f))
                
                # 对当月的 factor 做汇总
                month_factor_mean = df_model[column_name].mean()
                total_A.append(month_factor_mean)
                # print(f"monthly mean: {month_factor_mean}")

    return np.mean(total_A)

def compute_annual_sum_from_month(csv_folder, year, factor):
    total_A = 0.0
    column_name = factor + "_month"
    # 假设文件命名: SOC_terms_2007_{month:02d}_timestep_{month}_River.csv
    for month in range(1, 13):
        month_str = f"{month:02d}"
        filename = rf"SOC_terms_{year}_{month_str}_timestep_(\d+)_River\.csv"
        file_list = os.listdir(csv_folder)
        for f in file_list:
            if re.match(filename, f):
                # 读取 CSV
                df_model = pd.read_csv(os.path.join(csv_folder, f))
                
                # 对当月的 factor 做汇总
                month_erosion_mean = df_model[column_name].mean()
                total_A += month_erosion_mean
                # print(f"monthly mean: {month_erosion_mean}")

    return total_A

if __name__ == "__main__":
    factors = ['C', 'K', 'LS', 'P', 'R']
    start_year = 2008
    end_year = 2018
    
    years_list = []
    A_valid_list = []
    A_model_list = []
    
    valid_factor_lists = []
    model_factor_lists = []
    
    for year in range(start_year, end_year+1, 2):
        valid_factor_mean_list = []
        model_factor_mean_list = []
        for factor in factors:
            if factor == 'R':
                resolution = "1000"
                extension = 'Year'
                file_name = f"{factor}{str(year)}_{resolution}.tif"
            elif factor == 'C':
                resolution = '300'
                extension = 'year'
                file_name = f"{factor}{str(year)}_{resolution}.tif"
            else:
                resolution = "300"
                extension = resolution
                file_name = f"{factor}_{resolution}.tif"
                
            raster_path = DATA_DIR / "RUSLE1992-2019" / f"{factor}_{extension}" / file_name
            shp_path = DATA_DIR / "Loess_Plateau_vector_border.shp"
            csv_path = OUTPUT_DIR / "Data"
            
            factor_mean = compute_mean_value(raster_path, shp_path)
            valid_factor_mean_list.append(factor_mean)
            
            if factor == 'R':
                factor_mean = compute_annual_sum_from_month(csv_path, year, factor + "_factor")
            else:
                factor_mean = compute_annual_mean_from_month(csv_path, year, factor + "_factor")
                
            model_factor_mean_list.append(factor_mean)
        A_valid = math.prod(valid_factor_mean_list)
        print(f"C = {valid_factor_mean_list[0]}")
        print(f"K = {valid_factor_mean_list[1]}")
        print(f"LS = {valid_factor_mean_list[2]}")
        print(f"P = {valid_factor_mean_list[3]}")
        print(f"R = {valid_factor_mean_list[4]}")
        A_model = compute_annual_sum_from_month(csv_path, year, 'E_t_ha')
        print(f"数据计算的黄土高原范围内 {year} 年度土壤侵蚀量的平均值: {A_valid}")
        print(f"RUSLE模型计算的 {year} 年度土壤侵蚀量平均值: {A_model}")
        years_list.append(year)
        A_valid_list.append(A_valid)
        A_model_list.append(A_model)
        
        valid_factor_lists.append(valid_factor_mean_list)
        model_factor_lists.append(model_factor_mean_list)
    
    errors = np.array(A_valid_list) - np.array(A_model_list)
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    print(f"\nTotal RMSE: {rmse}\n")
    
    plt.figure(figsize=(8, 5))
    plt.plot(years_list, A_valid_list, marker='o', label='A_valid', linestyle='-')
    plt.plot(years_list, A_model_list, marker='s', label='A_model', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('A')
    plt.title('RUSLE Validation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "A_Model_vs_A_Valid.png"))
    print("A Validation plot saved as: \"A_Model_vs_A_Valid.png\"")
    plt.show()
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))  # 建两个 row, 每行最多放3个子图
    axes = axes.flatten()  # 展开成一维数组，方便索引
    for i, factor_name in enumerate(factors):
        ax = axes[i]
        # 取出当前因子在所有年份上的验证值和模型值
        y_valid = [valid_factor_lists[year_idx][i] for year_idx in range(len(years_list))]
        y_model = [model_factor_lists[year_idx][i] for year_idx in range(len(years_list))]
        
        ax.plot(years_list, y_valid, marker='o', label='Validation')
        ax.plot(years_list, y_model, marker='s', label='Model')
        
        ax.set_title(f"Factor: {factor_name}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Factor Value")
        ax.legend()

    # 如果子图没有排满，可以把剩余子图隐藏，也可以留空
    if len(factors) < len(axes):
        for j in range(len(factors), len(axes)):
            axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "RUSLE_Factors_validation.png"))
    print("A Validation plot saved as: \"RUSLE_Factors_validation.png\"")
    plt.show()