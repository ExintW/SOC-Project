# -*- coding: utf-8 -*-
"""
ERA5 download
"""

import cdsapi
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

os.makedirs(DATA_DIR, exist_ok=True)

c = cdsapi.Client()

"""
Time range
"""
first_year = 1980
last_year = 1980
for year in range(first_year, last_year + 1):
    for month in range(1, 13):
        print("=========================================================")
        print(f"Downloading {year}-{month:02d}")
        c.retrieve(
            'reanalysis-era5-land',
            {
                # 'variable': ['friction_velocity'],
                'variable': [
                    "2m_dewpoint_temperature",
                    "2m_temperature",
                    "skin_temperature",
                    "soil_temperature_level_1",
                    "soil_temperature_level_2",
                    "soil_temperature_level_3",
                    "soil_temperature_level_4",
                    "total_precipitation",
                    "leaf_area_index_high_vegetation",
                    "leaf_area_index_low_vegetation"
                ],
                'year': str(year),
                'month': "{month:02d}".format(month=month),
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                "time": [
                    "00:00", "01:00", "02:00",
                    "03:00", "04:00", "05:00",
                    "06:00", "07:00", "08:00",
                    "09:00", "10:00", "11:00",
                    "12:00", "13:00", "14:00",
                    "15:00", "16:00", "17:00",
                    "18:00", "19:00", "20:00",
                    "21:00", "22:00", "23:00"
                ],
                'format': 'netcdf'
            },
            DATA_DIR / "ERA5" / "{year}-{month:02d}.nc".format(year=year, month=month))
        