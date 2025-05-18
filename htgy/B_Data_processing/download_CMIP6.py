import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from scipy.interpolate import griddata

# Append parent directory to path to access globals
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

import requests
from pathlib import Path

# 构造镜像地址（不保证都可用）
mirrors = [
    # ESGF official HTTP mirror (some datasets are available via direct GET)
    "https://aims3.llnl.gov/thredds/fileServer/css03_data/CMIP6/HighResMIP/NCAR/CESM1-CAM5-SE-HR/"
    "hist-1950/r1i1p1f1/Lmon/lai/gn/v20200810/",
    "https://data.dkrz.de/thredds/fileServer/css03_data/CMIP6/HighResMIP/NCAR/CESM1-CAM5-SE-HR/"
    "hist-1950/r1i1p1f1/Lmon/lai/gn/v20200810/"
]

filename = "lai_Lmon_CESM1-CAM5-SE-HR_hist-1950_r1i1p1f1_gn_195001-201412.nc"
output_path = DATA_DIR / filename  # 替换为你的本地路径，如 Path(DATA_DIR)

def try_download():
    for base in mirrors:
        url = base + filename
        print(f"尝试下载：{url}")
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"✅ 下载成功：{output_path}")
            return
        except Exception as e:
            print(f"❌ 下载失败：{e}")

    print("\n🚫 所有镜像下载失败。请手动下载：")
    print(f"👉 手动访问：https://esgf-node.llnl.gov/search/cmip6/")
    print(f"🔍 搜索文件名：{filename}")
    print("并使用登录后生成的 wget 脚本下载。")

# 执行下载
try_download()

sys.exit()

# -------------------------
# Step 1: Open remote LAI dataset
# -------------------------
url = (
    "https://esgf-data1.llnl.gov/thredds/dodsC/css03_data/CMIP6/HighResMIP/"
    "NCAR/CESM1-CAM5-SE-HR/hist-1950/r1i1p1f1/Lmon/lai/gn/v20200810/"
    "lai_Lmon_CESM1-CAM5-SE-HR_hist-1950_r1i1p1f1_gn_195001-201412.nc"
)

ds = xr.open_dataset(url)

# Extract 1D lat/lon
lat_vals = ds['lat'].values
lon_vals = ds['lon'].values
print("读取成功：", ds)

# -------------------------
# Step 2: Load 1 km target grid points (Loess Plateau)
# -------------------------
csv_pts = Path(PROCESSED_DIR) / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
df_pts = pd.read_csv(csv_pts)
lons = df_pts["LON"].values
lats = df_pts["LAT"].values

# -------------------------
# Step 3: Interpolate and compute annual stats
# -------------------------
output_dir = Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled"
os.makedirs(output_dir, exist_ok=True)

def interp_and_collect_unstructured(ds, var_name, label, save_interp_fname):
    # Extract lat, lon, and LAI values
    lats_src = ds['lat'].values
    lons_src = ds['lon'].values
    lai_all  = ds[var_name].values  # shape: (time, lndgrid)

    # Remove invalid points (lat/lon == 9.969e+36 is missing value)
    mask = (lats_src < 1e35) & (lons_src < 1e35)
    lats_src = lats_src[mask]
    lons_src = lons_src[mask]
    lai_all  = lai_all[:, mask]  # now shape: (time, valid_points)

    points_src = np.column_stack([lons_src, lats_src])
    points_dst = np.column_stack([lons, lats])  # 1 km target points

    # Create interpolated array: shape (time, n_target_points)
    lai_interp_all = np.empty((lai_all.shape[0], points_dst.shape[0]), dtype=np.float32)
    for t in range(lai_all.shape[0]):
        lai_interp_all[t, :] = griddata(
            points_src, lai_all[t, :],
            points_dst, method='linear'
        )

    # Wrap into DataArray
    time_coords = ds['time'].values
    lai_interp = xr.DataArray(
        lai_interp_all,
        dims=["time", "points"],
        coords={"time": time_coords},
        name=var_name
    )

    # Save to NetCDF
    ds_interp = xr.Dataset({var_name: lai_interp})
    interp_path = output_dir / save_interp_fname
    ds_interp.to_netcdf(interp_path)
    print(f"→ Saved interpolated {label} to {interp_path.name}")

    # Compute annual stats
    rows = []
    for year, grp in ds_interp[var_name].groupby("time.year"):
        arr = grp.values.ravel()
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        mu = np.nanmean(arr)
        print(f"{label} {int(year)} → min={mn:.3f}, max={mx:.3f}, mean={mu:.3f}")
        rows.append((label, int(year), mn, mx, mu))

    ds_interp.close()
    return rows

# -------------------------
# Step 4: Process and save
# -------------------------
lai_rows = []
try:
    print(">>> Processing LAI")
    lai_rows = interp_and_collect_unstructured(
        ds,
        var_name="lai",
        label="LAI",
        save_interp_fname="resampled_lai_points_1950-2015.nc"
    )
except Exception as e:
    print(f"Error processing LAI: {e}")

# Save annual stats
stats_file = output_dir / "annual_LAI_stats_1950-2015.csv"
with open(stats_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["variable", "year", "min", "max", "mean"])
    for row in lai_rows:
        writer.writerow(row)
print(f"→ Annual stats saved to {stats_file.name}")


