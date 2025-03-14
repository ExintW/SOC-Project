import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *


# -------------------------------------------------------------------------
# 1) PATHS & INPUT FILES
# -------------------------------------------------------------------------
excel_file = DATA_DIR / "River_Basin_Points.xlsx"
output_file = PROCESSED_DIR / "River_Basin_Points_Updated.xlsx"

# Folder where the NetCDF files for each year are stored
nc_folder = DATA_DIR / "ERA5"

# CSV with initial SOC (ORGA) for 2007
loess_csv_path = DATA_DIR / "resampled_Loess_Plateau_1km.csv"

# List of sheet names to process
sheet_names = [
    "CaiJiaChuan-2022",
    "LiCha-2024",
    "LuoYuGou-2008",
    "WeiBei-2010",
    "WangMaoGou-2017"
]

# -------------------------------------------------------------------------
# 2) PREPARE 2007 SOC & LAI INTERPOLATION
# -------------------------------------------------------------------------
# 2A) Read CSV for 2007 SOC
loess_df = pd.read_csv(loess_csv_path)

# Extract arrays for interpolation from CSV
csv_lons = loess_df["LON"].values
csv_lats = loess_df["LAT"].values
csv_orga = loess_df["ORGA"].values  # initial SOC (g/kg)
csv_points = np.column_stack((csv_lons, csv_lats))

# 2B) Open the 2007 NetCDF for LAI
nc_2007_path = f"{nc_folder}\\resampled_2007.nc"
ds_2007 = xr.open_dataset(nc_2007_path)

# Example: average lai_lv over 'valid_time' dimension
lai_annual_2007 = ds_2007["lai_lv"].mean(dim="valid_time")

# Extract arrays for interpolation from 2007 NetCDF
nc_lons_2007 = ds_2007["longitude"].values
nc_lats_2007 = ds_2007["latitude"].values
lai_values_2007 = lai_annual_2007.values
nc_points_2007 = np.column_stack((nc_lons_2007, nc_lats_2007))

# Close the dataset (we already have what we need in memory)
ds_2007.close()

# -------------------------------------------------------------------------
# 3) PROCESS EACH SHEET: EXTRACT YEAR, INTERPOLATE LAI FOR THAT YEAR + 2007 SOC & LAI
# -------------------------------------------------------------------------
xls = pd.ExcelFile(excel_file)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for sheet in sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name=sheet)

        # Ensure we have X, Y columns
        if not all(col in df.columns for col in ["X", "Y"]):
            print(f"Sheet '{sheet}' missing 'X' or 'Y'. Skipping.")
            continue

        # Parse the measurement year from the sheet name (e.g., "CaiJiaChuan-2022" -> 2022)
        try:
            meas_year = int(sheet.split("-")[-1])
        except ValueError:
            print(f"Cannot parse year from sheet '{sheet}'. Skipping.")
            continue

        # -----------------------------------------------------------------
        # 3A) Interpolate LAI for the measurement year
        # -----------------------------------------------------------------
        nc_meas_path = f"{nc_folder}\\resampled_{meas_year}.nc"
        ds_meas = xr.open_dataset(nc_meas_path)

        # Average over valid_time dimension
        lai_annual_meas = ds_meas["lai_lv"].mean(dim="valid_time")

        # Extract arrays for interpolation
        nc_lons_meas = ds_meas["longitude"].values
        nc_lats_meas = ds_meas["latitude"].values
        lai_values_meas = lai_annual_meas.values
        nc_points_meas = np.column_stack((nc_lons_meas, nc_lats_meas))

        ds_meas.close()

        # Prepare target points from the Excel DataFrame
        target_points = df[["X", "Y"]].values

        # Interpolate measurement-year LAI
        lai_interp_meas = griddata(
            nc_points_meas,
            lai_values_meas,
            target_points,
            method="linear"
        )

        # Fill NaN with nearest if needed
        if np.any(np.isnan(lai_interp_meas)):
            nan_mask = np.isnan(lai_interp_meas)
            lai_interp_meas[nan_mask] = griddata(
                nc_points_meas,
                lai_values_meas,
                target_points[nan_mask],
                method="nearest"
            )

        # Add the measurement year LAI column
        df["LAI"] = lai_interp_meas

        # -----------------------------------------------------------------
        # 3B) Interpolate 2007 SOC from CSV
        # -----------------------------------------------------------------
        soc_2007 = griddata(
            csv_points,
            csv_orga,
            target_points,
            method="linear"
        )

        # Fill NaN with nearest if needed
        if np.any(np.isnan(soc_2007)):
            nan_mask = np.isnan(soc_2007)
            soc_2007[nan_mask] = griddata(
                csv_points,
                csv_orga,
                target_points[nan_mask],
                method="nearest"
            )

        df["2007 SOC (g/kg)"] = soc_2007

        # -----------------------------------------------------------------
        # 3C) Interpolate 2007 LAI from the NetCDF 2007
        # -----------------------------------------------------------------
        lai_2007 = griddata(
            nc_points_2007,
            lai_values_2007,
            target_points,
            method="linear"
        )

        # Fill NaN with nearest if needed
        if np.any(np.isnan(lai_2007)):
            nan_mask = np.isnan(lai_2007)
            lai_2007[nan_mask] = griddata(
                nc_points_2007,
                lai_values_2007,
                target_points[nan_mask],
                method="nearest"
            )

        df["2007 LAI"] = lai_2007

        # -----------------------------------------------------------------
        # 3D) Compute SOC Monthly Increase (g/kg/month)
        #      = (SOC(meas_year) - SOC(2007)) / ((meas_year - 2007)*12)
        # -----------------------------------------------------------------
        # We assume the sheet already has a column "SOC (g/kg)" for the measurement year
        if "SOC (g/kg)" in df.columns and meas_year > 2007:
            df["SOC Monthly Increase (g/kg/month)"] = (
                    (df["SOC (g/kg)"] - df["2007 SOC (g/kg)"])
                    / ((meas_year - 2007) * 12)
            )
        else:
            df["SOC Monthly Increase (g/kg/month)"] = np.nan

        # -----------------------------------------------------------------
        # Write updated sheet to the output Excel
        # -----------------------------------------------------------------
        df.to_excel(writer, sheet_name=sheet, index=False)

print("Done! The output file now includes:")
print("  - LAI (measurement year)")
print("  - 2007 SOC (g/kg)")
print("  - 2007 LAI")
print("  - SOC Monthly Increase (g/kg/month)")
