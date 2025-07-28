# -*- coding: utf-8 -*-
"""
ERA5 download
"""

import cdsapi

c = cdsapi.Client()

"""
Time range
"""
first_year = 1950
last_year = 2025
for year in range(first_year, last_year + 1):
        print("=========================================================")
        print("Downloading {year}".format(year=year))
        c.retrieve(
            "reanalysis-era5-land-monthly-means",
            {
                'product_type': ["monthly_averaged_reanalysis"],
                'variable': [
                    #"2m_dewpoint_temperature",
                    #"2m_temperature",
                    "skin_temperature",
                    "soil_temperature_level_1",
                    "soil_temperature_level_2",
                    #"soil_temperature_level_3",
                    #"soil_temperature_level_4",
                    #"total_precipitation",
                    #"leaf_area_index_high_vegetation",
                    #"leaf_area_index_low_vegetation"
                ],
                'year': str(year),
                'month': [
                    "01", "02", "03",
                    "04", "05", "06",
                    "07", "08", "09",
                    "10", "11", "12"
                        ],
                'time': ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived"
            },
            rf"D:\EcoSci\Dr.Shi\SOC_Github\Raw_Data\ERA5_Temp\{year}.nc"
        )