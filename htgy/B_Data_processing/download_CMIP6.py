import os
import sys
import xarray as xr

# Append parent directory to path to access 'globals' if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from globals import *  # Expects DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

url = (
    "http://aims3.llnl.gov/thredds/dodsC/css03_data/CMIP6/CMIP/"
    "BCC/BCC-CSM2-MR/historical/r1i1p1f1/Lmon/lai/gn/v20181114/"
    "lai_Lmon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc"
)
ds = xr.open_dataset(url)

# 查看原始范围
print("原始纬度范围:", ds.lat.min().item(), ds.lat.max().item())
print("原始经度范围:", ds.lon.min().item(), ds.lon.max().item())

# 切片
ds_sub = ds.sel(
    time = slice("1950-01-01", "2014-12-31"),
    lat  = slice(33, 41),    # 注意升序
    lon  = slice(103, 113),
)

print(ds_sub)  # 确认 lat, lon 都不是 0
ds_sub.to_netcdf(DATA_DIR / "LAI_LoessPlateau_195001-201412.nc")
print("子集保存完毕")
