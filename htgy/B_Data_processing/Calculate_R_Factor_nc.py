import xarray as xr
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

ERA5_path = DATA_DIR / "ERA5"
output_path = PROCESSED_DIR / "ERA5_R_Factor_Data"

first_year = 2007
last_year = 2007

for year in range(first_year, last_year+1):
    for month in range(1, 13):
        file_path = ERA5_path / f'{year}-{month:02d}.nc'
        output_file = output_path / f"{year}-{month:02d}_R_factor.nc"

        ds_hourly = xr.open_mfdataset(file_path, combine='by_coords', engine='netcdf4')

        ds_daily = ds_hourly.resample(valid_time="1D").sum()  # sum over each 24-hour block

        # convert to mm：
        ds_daily['tp'] = ds_daily['tp'] * 1000.0
        ds_daily['tp'].attrs['units'] = "mm/day"
        
        tp_daily = ds_daily['tp']
        rain9_daily = tp_daily.where(tp_daily >= 9.0, other=0.0)
        rain9_sum = rain9_daily.sum(dim="valid_time")
        
        R_factor = 8.3462 * (rain9_sum ** 1.25709)
        
        time_label = f"{year}-{month:02d}"
        R_factor = R_factor.expand_dims(time=[time_label])
        
        ds_out = R_factor.to_dataset(name="R_factor")
        
        ds_out["R_factor"].attrs["long_name"] = (
            "Monthly R factor from daily precipitation >= 9 mm"
        )
        ds_out["R_factor"].attrs["units"] = "?"
        
        ds_out.to_netcdf(output_file)
        print(f"R_factor saved to {output_file}")
