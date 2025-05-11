# -*- coding: utf-8 -*-
"""
avhrr download + crop to Loess Plateau border
（保留完整规则网格，在边界外仅用 NaN 填充）
"""

import os
import sys
import warnings
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: 注册 .rio accessor

# 让 globals.py 中的 DATA_DIR 可用
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import DATA_DIR

# 配置
BASE_URL    = "https://www.ncei.noaa.gov/data/land-leaf-area-index-and-fapar/access"
OUT_ROOT    = DATA_DIR / "AVHRR"
START_YEAR  = 1982
END_YEAR    = 2025
BORDER_PATH = DATA_DIR / "Loess_Plateau_vector_border.shp"

# 读取并投影边界
border_gdf    = gpd.read_file(BORDER_PATH).to_crs("EPSG:4326")
border_geoms  = list(border_gdf.geometry)
minx, miny, maxx, maxy = border_gdf.total_bounds  # [west, south, east, north]

# 开启 HTTP session
session = requests.Session()

# 全局抑制那条 cast 警告
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
)

for year in range(START_YEAR, END_YEAR + 1):
    year_url       = f"{BASE_URL}/{year}/"
    local_year_dir = OUT_ROOT / str(year)
    local_year_dir.mkdir(parents=True, exist_ok=True)

    print(f"Year {year}: listing {year_url}")
    resp     = session.get(year_url); resp.raise_for_status()
    soup     = BeautifulSoup(resp.text, "html.parser")
    nc_files = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".nc")]

    for fname in nc_files:
        tmp_path = local_year_dir / f"tmp_{fname}"
        out_path = local_year_dir / fname
        file_url = year_url + fname

        if out_path.exists():
            continue

        # 1) 下载整个 .nc 到临时文件
        print(f"  ↓ downloading {fname}")
        with session.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        # 2) 打开、删除边界变量、设置 CRS/维度，并 load 进内存
        with xr.open_dataset(tmp_path) as ds:
            # 删除后缀为 bnds 的那些变量
            drop_vars = [v for v in ds.data_vars if v.endswith("bnds")]
            if drop_vars:
                ds = ds.drop_vars(drop_vars)

            # 探测 lon/lat 维度名
            x_dim = "longitude" if "longitude" in ds.dims else "lon"
            y_dim = "latitude"  if "latitude"  in ds.dims else "lat"

            # 写入 CRS 并指定空间维度
            ds = ds.rio.write_crs("EPSG:4326", inplace=True)
            ds = ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)

            # load 进内存，确保文件句柄关闭
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ds.load()

            # 3) 先按边界框裁剪一次，加速后续多边形裁剪
            lat0, lat1 = ds[y_dim].values[0], ds[y_dim].values[-1]
            if lat0 > lat1:
                ds = ds.sel({ x_dim: slice(minx, maxx),
                              y_dim: slice(maxy, miny) })
            else:
                ds = ds.sel({ x_dim: slice(minx, maxx),
                              y_dim: slice(miny, maxy) })

            # 4) 精确多边形裁剪，但保留完整网格(drop=False)
            print(f"    ✂ cropping to Loess Plateau (keep grid)")
            clipped = ds.rio.clip(border_geoms, border_gdf.crs, drop=False)

            # 5) 输出裁剪结果
            clipped.to_netcdf(out_path)
            clipped.close()

        # 6) 删除临时文件
        tmp_path.unlink()

    print(f"⇾ done {year}\n")

print("All years downloaded & cropped (full grid preserved).")
