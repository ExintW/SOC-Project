import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import os
import sys
import geopandas as gpd
from shapely.geometry import Point

# allow imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# -------------------------------------------------------------------------
# 1) PATHS & INPUT FILES
# -------------------------------------------------------------------------
input_csv      = DATA_DIR     / "Vege_Input_Data.csv"
loess_csv      = PROCESSED_DIR / "resampled_Loess_Plateau_1km.csv"
border_shp     = DATA_DIR     / "Loess_Plateau_vector_border.shp"
output_csv     = PROCESSED_DIR / "Vege_Input_Data_CMIP6.csv"
cmip6_lai_path = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2001-2014.nc"

# -------------------------------------------------------------------------
# 2) READ & FILTER: VEGETATION INPUT CSV & BORDER
# -------------------------------------------------------------------------
df = pd.read_csv(input_csv, encoding='ISO-8859-1')
for col in ["Longitude", "Latitude", "SOC (kg C m^-2)", "Investigation time"]:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in {input_csv}")

# spatial filter to Loess Plateau
df_geo = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
border = gpd.read_file(border_shp)
if border.crs:
    df_geo = df_geo.set_crs(border.crs, allow_override=True)
union = border.geometry.unary_union
df_geo = df_geo[df_geo.within(union)]
df = pd.DataFrame(df_geo.drop(columns="geometry"))

# extract investigation year for SOC‐rate calculation
df["Investigation_Year"] = df["Investigation time"].astype(int)

# convert SOC units: kg C/m² → g C per kg soil
bulk_density = 1300  # kg/m³
depth         = 0.2   # m
soil_mass     = bulk_density * depth  # kg soil per m²
df["SOC (g/kg)"] = df["SOC (kg C m^-2)"] * (1000.0 / soil_mass)

# -------------------------------------------------------------------------
# 3) READ & PREPARE: LOESS SOC & CMIP6 2007 LAI
# -------------------------------------------------------------------------
# 3A) Loess CSV for 2007 SOC
loess_df = pd.read_csv(loess_csv)
for c in ["LON", "LAT", "ORGA"]:
    if c not in loess_df.columns:
        raise ValueError("Loess CSV must contain 'LON','LAT','ORGA'")
loess_pts  = np.column_stack((loess_df.LON.values, loess_df.LAT.values))
loess_orga = loess_df.ORGA.values  # 2007 SOC in g/kg

# 3B) CMIP6 monthly LAI (2001–2014) → compute 2007 annual mean
ds_lai        = xr.open_dataset(cmip6_lai_path)
lai_2007_grid = ds_lai["lai"].sel(time=ds_lai.time.dt.year == 2007).mean(dim="time").values
lons, lats    = ds_lai["lon"].values, ds_lai["lat"].values
grid_pts      = np.column_stack((lons, lats))
ds_lai.close()

# -------------------------------------------------------------------------
# 4) INTERPOLATE: 2007 SOC & 2007 LAI
# -------------------------------------------------------------------------
pts = df[["Longitude", "Latitude"]].values

# 4A) 2007 SOC interpolation (Loess → points)
soc2007 = griddata(loess_pts, loess_orga, pts, method="linear")
mask = np.isnan(soc2007)
if mask.any():
    soc2007[mask] = griddata(loess_pts, loess_orga, pts[mask], method="nearest")
df["2007 SOC (g/kg)"] = soc2007

# 4B) 2007 LAI interpolation (CMIP6 → points)
lai2007 = griddata(grid_pts, lai_2007_grid, pts, method="linear")
mask = np.isnan(lai2007)
if mask.any():
    lai2007[mask] = griddata(grid_pts, lai_2007_grid, pts[mask], method="nearest")
df["2007 LAI"] = lai2007

# -------------------------------------------------------------------------
# 5) CALCULATE MONTHLY SOC INCREASE
# -------------------------------------------------------------------------
def monthly_rate(row):
    yr      = row["Investigation_Year"]
    soc_i   = row["SOC (g/kg)"]
    soc_07  = row["2007 SOC (g/kg)"]
    if pd.isna(soc_i) or pd.isna(soc_07):
        return np.nan
    if yr == 2007:
        return 0.0
    dy = abs(yr - 2007)
    if yr > 2007:
        return (soc_i   - soc_07) / (dy * 12)
    else:
        return (soc_07 - soc_i ) / (dy * 12)

df["SOC Monthly Increase (g/kg/month)"] = df.apply(monthly_rate, axis=1)

# -------------------------------------------------------------------------
# 6) SAVE UPDATED TABLE
# -------------------------------------------------------------------------
df.to_csv(output_csv, index=False)
print(f"Done! Updated file saved to: {output_csv}")
