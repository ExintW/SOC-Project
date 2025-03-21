import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import os
import sys
import geopandas as gpd
from shapely.geometry import Point

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# -------------------------------------------------------------------------
# 1) PATHS & INPUT FILES
# -------------------------------------------------------------------------
nc_folder = DATA_DIR / "ERA5"  # Folder containing the NetCDF files (resampled_XXXX.nc)

# Input CSV (single sheet) with vegetation data
input_csv = DATA_DIR / "Vege_Input_Data.csv"

# CSV with 2007 SOC from Loess Plateau (used for 2007 SOC interpolation)
loess_csv_path = DATA_DIR / "resampled_Loess_Plateau_1km.csv"

# Shapefile with the Loess Plateau border
border_shp = DATA_DIR / "Loess_Plateau_vector_border.shp"

# Output CSV with the new columns
output_csv = PROCESSED_DIR / "Vege_Input_Data_Updated.csv"

# -------------------------------------------------------------------------
# 2) READ & FILTER: VEGETATION INPUT CSV & BORDER
# -------------------------------------------------------------------------
# 2A) Read the vegetation CSV
df = pd.read_csv(input_csv, encoding='ISO-8859-1')

# Check that the expected columns exist.
# For this updated file, we assume the columns are:
# "Longitude", "Latitude", "SOC (kg C m^-2)", and "Investigation time"
required_cols = ["Longitude", "Latitude", "SOC (kg C m^-2)", "Investigation time"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in {input_csv}")

# 2B) Convert the CSV points to a GeoDataFrame
df_geo = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude'])
)

# 2C) Read the border shapefile
border_gdf = gpd.read_file(border_shp)
# If the border's CRS is defined, assign (or transform) the GeoDataFrame to that CRS.
if border_gdf.crs:
    df_geo = df_geo.set_crs(border_gdf.crs, allow_override=True)
    # Alternatively, if your CSV is in a different CRS, reproject it:
    # df_geo = df_geo.to_crs(border_gdf.crs)

# 2D) Filter to include only points within the border
# The unary_union ensures all border polygons are combined into one geometry.
border_union = border_gdf.geometry.union_all()
df_geo = df_geo[df_geo.within(border_union)]

# Optionally drop the geometry column if you don't need it later.
df = pd.DataFrame(df_geo.drop(columns='geometry'))

# Convert "Investigation time" to integer years (if not already)
df["Investigation_Year"] = df["Investigation time"].astype(int)

# 2E) Convert SOC from kg C/m² to g/kg soil using:
#     SOC (g/kg soil) = (SOC (kg C/m²) / (bulk_density * depth)) * 1000
# Example: For SOC = 2.5 kg C/m², bulk density = 1300 kg/m³, depth = 0.3 m:
#          mass of soil = 1300 * 0.3 = 390 kg/m²,
#          SOC (g/kg) = (2.5 / 390) * 1000 ≈ 6.41 g/kg
bulk_density = 1300  # kg/m³
depth = 0.3          # m
mass_of_soil = bulk_density * depth  # e.g., 390 kg/m²
df["SOC (g/kg)"] = df["SOC (kg C m^-2)"] * (1000 / mass_of_soil)

# -------------------------------------------------------------------------
# 3) READ & PREPARE: LOESS SOC & 2007 LAI
# -------------------------------------------------------------------------
# 3A) Read the Loess CSV for 2007 SOC interpolation
loess_df = pd.read_csv(loess_csv_path)
if not all(col in loess_df.columns for col in ["LON", "LAT", "ORGA"]):
    raise ValueError("Loess CSV must contain 'LON', 'LAT', and 'ORGA' (2007 SOC) columns.")

csv_lons = loess_df["LON"].values
csv_lats = loess_df["LAT"].values
csv_orga = loess_df["ORGA"].values  # 2007 SOC in g/kg soil
csv_points = np.column_stack((csv_lons, csv_lats))

# 3B) Read the NetCDF for 2007 LAI (file: resampled_2007.nc)
nc_2007_path = nc_folder / "resampled_2007.nc"
ds_2007 = xr.open_dataset(nc_2007_path)
# Average LAI over the 'valid_time' dimension
lai_annual_2007 = ds_2007["lai_lv"].mean(dim="valid_time")
nc_lons_2007 = ds_2007["longitude"].values
nc_lats_2007 = ds_2007["latitude"].values
lai_values_2007 = lai_annual_2007.values
nc_points_2007 = np.column_stack((nc_lons_2007, nc_lats_2007))
ds_2007.close()

# -------------------------------------------------------------------------
# 4) INTERPOLATE: 2007 SOC, 2007 LAI, & Investigation-year LAI
# -------------------------------------------------------------------------
# Create target points array from the vegetation CSV
points = df[["Longitude", "Latitude"]].values

# 4A) Interpolate 2007 SOC using the Loess CSV
soc_2007 = griddata(csv_points, csv_orga, points, method="linear")
# Fill NaN using nearest neighbor interpolation
nan_mask = np.isnan(soc_2007)
if np.any(nan_mask):
    soc_2007[nan_mask] = griddata(csv_points, csv_orga, points[nan_mask], method="nearest")
df["2007 SOC (g/kg)"] = soc_2007

# 4B) Interpolate 2007 LAI from the 2007 NetCDF
lai_2007 = griddata(nc_points_2007, lai_values_2007, points, method="linear")
nan_mask = np.isnan(lai_2007)
if np.any(nan_mask):
    lai_2007[nan_mask] = griddata(nc_points_2007, lai_values_2007, points[nan_mask], method="nearest")
df["2007 LAI"] = lai_2007

# 4C) Interpolate LAI for the Investigation Year for each record.
#     Group by unique investigation year to avoid opening a NetCDF multiple times.
df["LAI_investigation_year"] = np.nan  # placeholder column
unique_years = df["Investigation_Year"].unique()
for year in unique_years:
    mask = (df["Investigation_Year"] == year)
    sub_points = df.loc[mask, ["Longitude", "Latitude"]].values

    # Open the corresponding NetCDF for the investigation year
    nc_path = nc_folder / f"resampled_{year}.nc"
    if not nc_path.exists():
        df.loc[mask, "LAI_investigation_year"] = np.nan
        continue

    ds_meas = xr.open_dataset(nc_path)
    lai_annual_meas = ds_meas["lai_lv"].mean(dim="valid_time")
    ds_meas.close()

    nc_lons_meas = lai_annual_meas["longitude"].values
    nc_lats_meas = lai_annual_meas["latitude"].values
    lai_values_meas = lai_annual_meas.values
    nc_points_meas = np.column_stack((nc_lons_meas, nc_lats_meas))

    lai_interp = griddata(nc_points_meas, lai_values_meas, sub_points, method="linear")
    # Fill any NaN values using nearest neighbor interpolation
    nan_mask2 = np.isnan(lai_interp)
    if np.any(nan_mask2):
        lai_interp[nan_mask2] = griddata(nc_points_meas, lai_values_meas, sub_points[nan_mask2], method="nearest")

    df.loc[mask, "LAI_investigation_year"] = lai_interp

# -------------------------------------------------------------------------
# 5) CALCULATE MONTHLY SOC INCREASE
#
# For each record:
#   - If Investigation_Year > 2007:
#         (SOC (g/kg) [investigation] - 2007 SOC (g/kg)) / ((Investigation_Year - 2007) * 12)
#   - If Investigation_Year < 2007:
#         (2007 SOC (g/kg) - SOC (g/kg) [investigation]) / ((2007 - Investigation_Year) * 12)
#   - If Investigation_Year == 2007, set to 0.
# -------------------------------------------------------------------------
def monthly_soc_increase(row):
    year = row["Investigation_Year"]
    soc_invest = row["SOC (g/kg)"]
    soc_2007_val = row["2007 SOC (g/kg)"]

    if pd.isna(soc_invest) or pd.isna(soc_2007_val):
        return np.nan
    if year == 2007:
        return 0.0
    elif year > 2007:
        return (soc_invest - soc_2007_val) / ((year - 2007) * 12)
    else:
        return (soc_2007_val - soc_invest) / ((2007 - year) * 12)

df["SOC Monthly Increase (g/kg/month)"] = df.apply(monthly_soc_increase, axis=1)

# -------------------------------------------------------------------------
# 6) SAVE THE UPDATED CSV
# -------------------------------------------------------------------------
df.to_csv(output_csv, index=False)
print(f"Done! Updated file saved to: {output_csv}")
