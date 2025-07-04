import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rioxarray         # ← pip install rioxarray
import xarray as xr
from scipy.ndimage import gaussian_filter   # ← pip install scipy
from scipy.interpolate import griddata      # ← pip install scipy
from pathlib import Path

# allow imports from parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# ─────────── File Paths ───────────
tiff_k1_path    = DATA_DIR    / "k1_halfDegree.tif"
tiff_k2_path    = DATA_DIR    / "k2_halfDegree.tif"
csv_file_path   = PROCESSED_DIR / "Resampled_Loess_Plateau_1km_with_DEM_region_labeled.csv"
output_csv_path = PROCESSED_DIR / "Resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
os.makedirs(output_csv_path.parent, exist_ok=True)

# ─────────── Helper: two‐step interpolation with negative‐masking ───────────
def interp_tiff(path, lons, lats):
    da = rioxarray.open_rasterio(path)
    nodata = da.rio.nodata
    da = da.where((da != nodata) & (da >= 0.0))
    da = da.squeeze("band", drop=True).rename({"x":"lon","y":"lat"})
    da_lin = da.interp(lon=xr.DataArray(lons, dims="points"),
                      lat=xr.DataArray(lats, dims="points"),
                      method="linear")
    da_nn  = da.interp(lon=xr.DataArray(lons, dims="points"),
                      lat=xr.DataArray(lats, dims="points"),
                      method="nearest")
    return da_lin.fillna(da_nn).values

# ─────────── SOM→SOC（月率）转换 ───────────
def convert_som_to_soc_monthly(som_k_day):
    k_day = np.maximum(som_k_day, 0.0)
    return (1 - np.exp(-k_day * 30)) * 0.58

# ─────────── 读入 CSV & 插值 ───────────
df = pd.read_csv(csv_file_path)
lon_csv = df["LON"].values
lat_csv = df["LAT"].values

som_k1_day = interp_tiff(tiff_k1_path, lon_csv, lat_csv)
som_k2_day = interp_tiff(tiff_k2_path, lon_csv, lat_csv)

# ─────────── 计算 SOC 速率并写入 DataFrame ───────────
df["SOC_k1_fast_pool (1/day)"]   = som_k1_day * 0.58
df["SOC_k2_slow_pool (1/day)"]   = som_k2_day * 0.58
df["SOC_k1_fast_pool (1/month)"] = convert_som_to_soc_monthly(som_k1_day)
df["SOC_k2_slow_pool (1/month)"] = convert_som_to_soc_monthly(som_k2_day)

# ─────────── 对月率数据做“先填洞”+Gaussian 平滑 ───────────
lon_vals = np.unique(lon_csv)
lat_vals = np.unique(lat_csv)

for col in ["SOC_k1_fast_pool (1/month)", "SOC_k2_slow_pool (1/month)"]:
    # 1) pivot 成规则网格（行：LAT，列：LON）
    grid = df.pivot(index="LAT", columns="LON", values=col)
    grid = grid.sort_index(ascending=True).reindex(sorted(grid.columns), axis=1)
    arr = grid.values

    # 2) 先用 nearest‐neighbour 填充 NaN 空洞
    #    构造所有点的经纬坐标对 pts，valid 标记非 NaN 的点
    lons_mesh, lats_mesh = np.meshgrid(grid.columns.values, grid.index.values)
    pts      = np.column_stack((lons_mesh.ravel(), lats_mesh.ravel()))
    vals     = arr.ravel()
    valid    = ~np.isnan(vals)
    filled   = griddata(pts[valid], vals[valid], pts, method="nearest")
    arr_filled = filled.reshape(arr.shape)

    # 3) 对填充后的矩阵做 Gaussian 平滑
    sigma = 20.0   # 可调整平滑强度
    smoothed = gaussian_filter(arr_filled, sigma=sigma)

    # 4) 映射回原始 DataFrame
    smooth_df = pd.DataFrame(smoothed, index=grid.index, columns=grid.columns)
    df[col] = df.apply(lambda r: smooth_df.at[r["LAT"], r["LON"]], axis=1)

# ─────────── 保存 CSV 并绘图 ───────────
df.to_csv(output_csv_path, index=False)
print(f"✅ 平滑后 CSV 已保存：{output_csv_path}")

for col, label, fname in [
    ("SOC_k1_fast_pool (1/month)", "k₁ fast‐pool rate (1/month)", "map_k1_fast_pool.png"),
    ("SOC_k2_slow_pool (1/month)", "k₂ slow‐pool rate (1/month)", "map_k2_slow_pool.png"),
]:
    fig, ax = plt.subplots(figsize=(10,6))
    sc = ax.scatter(df["LON"], df["LAT"], c=df[col], s=10, edgecolor="none")
    plt.colorbar(sc, ax=ax, label=label)
    ax.set(xlabel="Longitude", ylabel="Latitude",
           xlim=(lon_csv.min(), lon_csv.max()),
           ylim=(lat_csv.min(), lat_csv.max()),
           title=f"Spatial distribution of {label}")
    ax.margins(0)
    plt.tight_layout()
    plt.savefig(output_csv_path.parent / fname)
    plt.close(fig)
    print(f"📊 平滑地图已保存：{fname}")
