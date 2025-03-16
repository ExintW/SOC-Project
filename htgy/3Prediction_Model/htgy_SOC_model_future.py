import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import netCDF4 as nc
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import sys

# Append the parent directory and import globals (DATA_DIR, PROCESSED_DIR, OUTPUT_DIR)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# =============================================================================
# 0) SETUP: READ THE LOESS PLATEAU BORDER SHAPEFILE
# =============================================================================
# Read the Loess Plateau border shapefile and combine all features into one geometry.
loess_border_path = DATA_DIR / "Loess_Plateau_vector_border.shp"
loess_border = gpd.read_file(loess_border_path)
# Use union_all() to merge all features (recommended over the deprecated unary_union).
loess_border_geom = loess_border.union_all()

# =============================================================================
# 1) LOAD FINAL SOC FROM 2025 & SETUP DAM + PROPORTION DATA
# =============================================================================
# (A) Load the final SOC from your previous run (2007-2025):
final_csv_path = OUTPUT_DIR / "Data" / "SOC_terms_2025_01_timestep_217.csv"
df_init = pd.read_csv(final_csv_path, encoding='utf-8-sig')

# (B) Dam CSV & proportion CSV remain the same:
dam_csv_path = PROCESSED_DIR / "htgy_Dam_with_matched_points.csv"
proportion_csv_path = DATA_DIR / "Fast_Slow_SOC_Proportion.csv"

df_dam = pd.read_csv(dam_csv_path, encoding='utf-8-sig')
df_prop = pd.read_csv(proportion_csv_path, encoding='utf-8-sig')

# Ensure the 'year' column (dam construction year) is numeric
df_dam["year"] = pd.to_numeric(df_dam["year"], errors="coerce")
df_dam["total_stor"] = pd.to_numeric(df_dam["total_stor"], errors="coerce")
df_dam["deposition"] = pd.to_numeric(df_dam["deposition"], errors="coerce")
df_dam["capacity_remained"] = df_dam["total_stor"] - df_dam["deposition"]

# Column names from your region CSV (used below)
lon_col, lat_col = "LON", "LAT"
dem_col = "htgy_DEM"
slope_col = "SLOPE"
landuse_col = "LANDUSE"
k1_col = "SOC_k1_fast_pool (1/month)"
k2_col = "SOC_k2_slow_pool (1/month)"

# -----------------------------------------------------------------------------
# Helper: pivot_2d
# -----------------------------------------------------------------------------
def pivot_2d(df_in, value_col, lat_col="LAT", lon_col="LON"):
    """
    Pivot a CSV with columns [lat_col, lon_col, value_col] into a 2D array.
    Sorted so that lat is descending and lon ascending.
    """
    return (df_in.pivot(index=lat_col, columns=lon_col, values=value_col)
             .sort_index(ascending=False)
             .values)

# Extract the grid coordinates from the final CSV (which has LAT, LON columns)
grid_x = np.sort(df_init[lon_col].unique())
grid_y = np.sort(df_init[lat_col].unique())[::-1]

# Pivot out the final C_fast and C_slow as the new initial conditions (2025 state)
C_fast_init = pivot_2d(df_init, "C_fast", lat_col, lon_col)
C_slow_init = pivot_2d(df_init, "C_slow", lat_col, lon_col)

# =============================================================================
# 2) RELOAD REGION CSV (FOR DEM, SLOPE, LANDUSE, DECAY RATES)
# =============================================================================
region_csv_path = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
df_region = pd.read_csv(region_csv_path, encoding='utf-8-sig')

def create_grid(data, col_name):
    return (data.pivot(index=lat_col, columns=lon_col, values=col_name)
                .sort_index(ascending=False)
                .values)

DEM = create_grid(df_region, dem_col)
SLOPE = create_grid(df_region, slope_col)
LANDUSE = create_grid(df_region, landuse_col)
K_fast = create_grid(df_region, k1_col)
K_slow = create_grid(df_region, k2_col)

# Fill missing values with the mean.
DEM = np.nan_to_num(DEM, nan=np.nanmean(DEM))
SLOPE = np.nan_to_num(SLOPE, nan=np.nanmean(SLOPE))
K_fast = np.nan_to_num(K_fast, nan=np.nanmean(K_fast))
K_slow = np.nan_to_num(K_slow, nan=np.nanmean(K_slow))

# =============================================================================
# 2a) COMPUTE CONSTANT p_fast_grid FROM PROPORTION CSV (as in original code)
# =============================================================================
def allocate_fast_slow_soc(C, LANDUSE, proportion_df):
    """
    Partition total SOC (C) into fast and slow pools using land-use based percentages.
    The percentages are provided in the proportion CSV (df_prop).
    """
    prop_dict = {
        row['Type']: {
            'fast': row['Fast SOC(%)'] / 100,
            'slow': row['Slow SOC(%)'] / 100
        } for _, row in proportion_df.iterrows()
    }
    rows, cols = LANDUSE.shape
    C_fast_dummy = np.zeros((rows, cols))
    C_slow_dummy = np.zeros((rows, cols))
    p_fast_grid = np.zeros((rows, cols))  # Fraction of SOC in the fast pool.
    for i in range(rows):
        for j in range(cols):
            land_type = LANDUSE[i, j]
            # If land type is not found, assume all SOC goes to the slow pool.
            props = prop_dict.get(land_type, {'fast': 0, 'slow': 1})
            C_fast_dummy[i, j] = props['fast']  # Not using C here; we only need the fraction.
            C_slow_dummy[i, j] = props['slow']
            p_fast_grid[i, j] = props['fast']
    return C_fast_dummy, C_slow_dummy, p_fast_grid

# Compute the constant p_fast_grid from LANDUSE and df_prop.
_, _, p_fast_grid_const = allocate_fast_slow_soc(C_fast_init, LANDUSE, df_prop)

# =============================================================================
# 3) DAM DATA PROCESSING
# =============================================================================
BULK_DENSITY = 1300  # t/m³

def find_nearest_index(array, value):
    """Return the index of the element in 'array' closest to 'value'."""
    return (np.abs(array - value)).argmin()

# =============================================================================
# 4) RUSLE COMPONENTS
# =============================================================================
def calculate_r_factor_monthly(rain_month_mm):
    """
    Compute the monthly rainfall erosivity factor (R factor).

    For the Loess Plateau, studies (e.g., Zhao et al. 2012) suggest a coefficient
    about 4 times higher than the standard value, so:
        R = 6.94 * rain_month_mm
    """
    return 6.94 * rain_month_mm

def calculate_ls_factor(slope, slope_length=1000):
    """
    Compute the LS factor (slope length and steepness factor) from slope (in degrees).
    Uses the standard RUSLE formulation.
    """
    slope_rad = np.deg2rad(slope)
    return ((slope_length / 22.13)**0.4) * ((np.sin(slope_rad)/0.0896)**1.3)

def calculate_c_factor(lai):
    """
    Compute the C factor from LAI using an exponential decay.
    For the Loess Plateau, using 1.7 as the exponent yields better matches.
    """
    return np.exp(-1.7 * lai)

def calculate_p_factor(landuse):
    """
    Return the support practice factor (P factor) based on land use.
    """
    p_values = {
        "sloping cropland": 0.4,
        "forestland": 0.5,
        "grassland": 0.5,
        "not used": 0.5,
        "terrace": 0.1,
        "dam field": 0.05
    }
    return p_values.get(str(landuse).lower(), 1.0)

K_factor = np.full_like(DEM, 0.03)
LS_factor = calculate_ls_factor(SLOPE)
P_factor = np.array([
    [calculate_p_factor(LANDUSE[i, j]) for j in range(LANDUSE.shape[1])]
    for i in range(LANDUSE.shape[0])
])

# =============================================================================
# 5) CONVERT SOIL LOSS TO SOC LOSS (g/kg/month)
# =============================================================================
def convert_soil_loss_to_soc_loss_monthly(E_t_ha_month, ORGA_g_per_kg, bulk_density=1300):
    """
    Convert soil loss (in t/ha/month) to SOC loss (g/kg/month).
    1 t/ha = 100 g/m².
    """
    E_g_m2_month = E_t_ha_month * 100.0
    soc_loss_g_m2_month = E_g_m2_month * (ORGA_g_per_kg / 1000.0) * bulk_density
    soc_loss_g_kg_month = soc_loss_g_m2_month / bulk_density
    return soc_loss_g_kg_month

# =============================================================================
# 6) ROUTE SOIL AND SOC WITH DAMS
# =============================================================================
def distribute_soil_and_soc_with_dams(E_tcell, S, DEM, dam_positions, grid_x, grid_y):
    """
    Route soil and its associated SOC arriving from upstream cells.

    Key points:
      - Only the soil/SOC arriving from upstream (inflow) is deposited in a cell.
      - The cell's own erosion is not deposited locally; it is immediately routed downstream.
      - For cells with a dam, upstream inflow is retained up to the dam's capacity;
        any excess is routed downstream.

    Parameters:
      E_tcell : 2D array (t/cell/month) of local soil loss.
      S       : 2D array (kg/cell/month) of local SOC erosion.
      DEM     : 2D array of cell elevations.
      dam_positions : list of dicts with keys 'i', 'j', 'capacity_remained_tons'.
      grid_x, grid_y: 1D arrays of cell center coordinates.

    Returns:
      D_soil  : 2D array (t/cell/month) of soil deposited (upstream inflow).
      D_soc   : 2D array (kg/cell/month) of SOC deposited (upstream inflow).
      inflow_soil, inflow_soc: 2D arrays of soil and SOC still flowing downstream.
    """
    rows, cols = DEM.shape
    dam_capacity_map = {(dam['i'], dam['j']): dam['capacity_remained_tons'] for dam in dam_positions}
    inflow_soil = np.zeros_like(E_tcell, dtype=float)
    inflow_soc = np.zeros_like(E_tcell, dtype=float)
    D_soil = np.zeros_like(E_tcell, dtype=float)
    D_soc = np.zeros_like(E_tcell, dtype=float)

    all_cells = [(i, j) for i in range(rows) for j in range(cols)]
    all_cells.sort(key=lambda c: DEM[c[0], c[1]], reverse=True)

    for (i, j) in all_cells:
        deposition_soil = max(inflow_soil[i, j], 0)
        deposition_soc = max(inflow_soc[i, j], 0)

        if (i, j) in dam_capacity_map:
            cap = dam_capacity_map[(i, j)]
            if deposition_soil <= cap:
                D_soil[i, j] = deposition_soil
                D_soc[i, j] = deposition_soc
                dam_capacity_map[(i, j)] = cap - deposition_soil
                excess_soil = 0
                excess_soc = 0
            else:
                D_soil[i, j] = cap
                deposited_soc = (cap / deposition_soil) * deposition_soc if deposition_soil > 0 else 0
                deposited_soc = max(0, deposited_soc)
                D_soc[i, j] = deposited_soc
                dam_capacity_map[(i, j)] = 0
                excess_soil = deposition_soil - cap
                excess_soc = deposition_soc - deposited_soc
            current_inflow_soil = excess_soil
            current_inflow_soc = excess_soc
        else:
            D_soil[i, j] = deposition_soil
            D_soc[i, j] = deposition_soc
            current_inflow_soil = deposition_soil
            current_inflow_soc = deposition_soc

        source_soil = E_tcell[i, j]
        source_soc = S[i, j]

        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if DEM[ni, nj] < DEM[i, j]:
                        slope_diff = (DEM[i, j] - DEM[ni, nj]) / np.hypot(di, dj)
                        neighbors.append(((ni, nj), slope_diff))
        total_slope = sum(s for _, s in neighbors)
        if total_slope > 0:
            for ((ni, nj), slope_val) in neighbors:
                fraction = slope_val / total_slope
                inflow_soil[ni, nj] += source_soil * fraction
                inflow_soc[ni, nj] += source_soc * fraction
    return D_soil, D_soc, inflow_soil, inflow_soc

# =============================================================================
# 7) VEGETATION INPUT & UPDATED SOC DYNAMIC MODEL
# =============================================================================
def vegetation_input(LAI):
    """
    Compute vegetation input (g/kg/month) from LAI using an empirical formula.
    """
    return 0.00008128 * (LAI ** 7.33382537)

def soc_dynamic_model(C_fast, C_slow,
                      soc_loss_g_kg_month, D_soil, D_soc, V,
                      K_fast, K_slow, p_fast_grid, dt, M_soil):
    """
    Update the SOC pools (g/kg) for one month.

    Deposited SOC (D_soc, in kg) is converted to a concentration increment (g/kg)
    using: deposition_concentration = (D_soc * 1000) / M_soil.
    """
    erosion_fast = -soc_loss_g_kg_month * p_fast_grid
    erosion_slow = -soc_loss_g_kg_month * (1 - p_fast_grid)

    dep_conc = (D_soc * 1000.0) / M_soil
    dep_fast = dep_conc * p_fast_grid
    dep_slow = dep_conc * (1 - p_fast_grid)

    veg_fast = V * p_fast_grid
    veg_slow = V * (1 - p_fast_grid)

    reaction_fast = -K_fast * C_fast
    reaction_slow = -K_slow * C_slow

    C_fast_new = np.maximum(C_fast + (erosion_fast + dep_fast + veg_fast + reaction_fast) * dt, 0)
    C_slow_new = np.maximum(C_slow + (erosion_slow + dep_slow + veg_slow + reaction_slow) * dt, 0)
    return C_fast_new, C_slow_new

# =============================================================================
# 8) HELPER: REGRID CMIP6 POINT DATA TO 2D GRID
# =============================================================================
def create_grid_from_points(lon_points, lat_points, values, grid_x, grid_y):
    """
    Regrid 1D point data to a 2D grid by nearest coordinate match.
    """
    grid = np.full((len(grid_y), len(grid_x)), np.nan)
    for k in range(len(values)):
        j = (np.abs(grid_x - lon_points[k])).argmin()
        i = (np.abs(grid_y - lat_points[k])).argmin()
        grid[i, j] = values[k]
    return grid

# =============================================================================
# 9) FIGURE OUTPUT SETUP & PLOTTING INITIAL CONDITION
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot and save the initial SOC distribution (starting in 2025).
fig, ax = plt.subplots()
cax = ax.imshow(C_fast_init + C_slow_init, cmap="viridis",
                extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
cbar = fig.colorbar(cax, label="SOC (g/kg)")
ax.set_title("Initial SOC Distribution at 2025 (t = 0)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
ax.ticklabel_format(style='plain', axis='x')
plt.savefig(os.path.join(OUTPUT_DIR, "SOC_initial_2025.png"))
plt.close(fig)

# =============================================================================
# 10) MAIN SIMULATION LOOP (MONTHLY, 2025-2100)
# =============================================================================
CELL_AREA_HA = 100.0  # 1 km² = 100 ha
start_year = 2025
end_year = 2100
global_timestep = 0

# =============================================================================
# M_soil: Total soil mass per cell (kg).
# =============================================================================
# Soil loss is often reported in terms of mass loss per area.
# For example, a typical soil loss rate on the Loess Plateau is around 1000 t/ha/year.
#
# Convert 1000 t/ha/year to kg/m²:
#    1000 t/ha = 1000 * 1000 kg / 10,000 m² = 100 kg/m² per year.
#
# To convert this mass loss into a depth, we use the soil bulk density.
# If we assume a bulk density of 1300 kg/m³, then the soil loss depth is:
#    depth = (100 kg/m²) / (1300 kg/m³) ≈ 0.0769 m (or about 7.7 cm/year).
#
# In our model, we assume that the active soil layer that interacts with SOC
# (for deposition and dynamic exchange) is roughly equal to this soil loss depth.
#
# Therefore, for a 1 km² grid cell (1,000,000 m²), the total soil mass (M_soil)
# is calculated as:
#    M_soil = Area * effective_depth * bulk_density
#           = 1,000,000 m² * 0.077 m * 1300 kg/m³ ≈ 1.0e8 kg
#
# We update M_soil to 1.0e8 (instead of 3.9e8, which assumes a 30 cm depth).
M_soil = 1.0e8

# Initialize current SOC pools with the final SOC from 2025.
C_fast_current = C_fast_init.copy()
C_slow_current = C_slow_init.copy()

# ---------------------------------------------------------------------------
# (A) Open the CMIP6 NetCDF files ONCE
# ---------------------------------------------------------------------------
lai_file = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_2015-2100_126.nc"
pr_file  = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_pr_2015-2100_126.nc"

# Assume these files have a 'time' dimension from Jan 2015 to Dec 2100.
base_year = 2015  # The first year in these NetCDF files

with nc.Dataset(lai_file) as ds_lai, nc.Dataset(pr_file) as ds_pr:
    # LAI file variables
    lon_nc_lai = ds_lai.variables['lon'][:]  # Adjusted variable name if needed
    lat_nc_lai = ds_lai.variables['lat'][:]
    lai_data = ds_lai.variables['lai'][:]      # shape: (time, n_points)

    # Precipitation file variables
    lon_nc_pr = ds_pr.variables['lon'][:]
    lat_nc_pr = ds_pr.variables['lat'][:]
    pr_data = ds_pr.variables['pr'][:]         # shape: (time, n_points), in kg m^-2 s^-1

    # Yearly loop: 2025 -> 2100
    for year in range(start_year, end_year + 1):
        # Filter only those dams that have been built on or before this year.
        df_dam_active = df_dam[df_dam["year"] <= year].copy()
        dam_positions_active = []
        for _, row in df_dam_active.iterrows():
            i_idx = find_nearest_index(grid_y, row["y"])
            j_idx = find_nearest_index(grid_x, row["x"])
            capacity_10000_m3 = row["capacity_remained"]
            capacity_tons = capacity_10000_m3 * 10000 * BULK_DENSITY
            dam_positions_active.append({
                'i': i_idx,
                'j': j_idx,
                'capacity_remained_tons': capacity_tons
            })

        # Monthly loop: 1..12
        for month_idx in range(12):
            # netCDF index (assuming time=0 is Jan 2015)
            netcdf_index = (year - base_year) * 12 + month_idx
            if netcdf_index < 0 or netcdf_index >= lai_data.shape[0]:
                continue

            # --------------------------
            # Regrid LAI from CMIP6 data
            # --------------------------
            lai_1d = lai_data[netcdf_index, :]
            LAI_2D = create_grid_from_points(lon_nc_lai, lat_nc_lai, lai_1d, grid_x, grid_y)
            LAI_2D = np.nan_to_num(LAI_2D, nan=np.nanmean(LAI_2D))

            # --------------------------
            # Regrid Precipitation from CMIP6 data
            # --------------------------
            pr_1d = pr_data[netcdf_index, :]
            # Convert from kg m^-2 s^-1 to mm/month:
            # 1 kg m^-2 = 1 mm water; multiply by seconds per day and ~30 days per month.
            days_in_month = 30
            seconds_per_day = 86400
            pr_1d_mm = pr_1d * seconds_per_day * days_in_month
            RAIN_2D = create_grid_from_points(lon_nc_pr, lat_nc_pr, pr_1d_mm, grid_x, grid_y)
            RAIN_2D = np.nan_to_num(RAIN_2D, nan=np.nanmean(RAIN_2D))

            # --------------------------
            # Compute RUSLE factors
            # --------------------------
            R_month = calculate_r_factor_monthly(RAIN_2D)
            C_factor_2D = calculate_c_factor(LAI_2D)
            E_t_ha_month = R_month * K_factor * LS_factor * C_factor_2D * P_factor
            E_tcell_month = E_t_ha_month * CELL_AREA_HA

            # --------------------------
            # Compute SOC erosion and SOC loss
            # --------------------------
            S = E_tcell_month * (C_fast_current + C_slow_current)
            SOC_loss_g_kg_month = convert_soil_loss_to_soc_loss_monthly(
                E_t_ha_month, (C_fast_current + C_slow_current)
            )

            # --------------------------
            # Route soil and SOC with dam effects
            # --------------------------
            D_soil, D_soc, inflow_soil, inflow_soc = distribute_soil_and_soc_with_dams(
                E_tcell_month, S, DEM, dam_positions_active, grid_x, grid_y
            )

            # --------------------------
            # Compute vegetation input
            # --------------------------
            V = vegetation_input(LAI_2D)

            # --------------------------
            # Update SOC pools using the dynamic model.
            # Use the constant p_fast_grid from the proportion CSV.
            # --------------------------
            C_fast_current, C_slow_current = soc_dynamic_model(
                C_fast_current, C_slow_current,
                SOC_loss_g_kg_month, D_soil, D_soc, V,
                K_fast, K_slow, p_fast_grid_const,
                dt=1,
                M_soil=M_soil
            )

            global_timestep += 1
            print(f"Completed future simulation for Year {year}, Month {month_idx+1}, Timestep={global_timestep}")

            # -------------------------------------------------------------------------
            # (A) SAVE FIGURE OUTPUT
            # -------------------------------------------------------------------------
            fig, ax = plt.subplots()
            cax = ax.imshow(C_fast_current + C_slow_current, cmap="viridis",
                            extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
            cbar = fig.colorbar(cax, label="SOC (g/kg)")
            ax.set_title(f"SOC at Timestep {global_timestep} (Year {year}, Month {month_idx+1})")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='x')
            filename_fig = f"SOC_{year}_{month_idx+1:02d}_timestep_{global_timestep}.png"
            plt.savefig(os.path.join(OUTPUT_DIR / "Figure", filename_fig))
            plt.close(fig)

            # -------------------------------------------------------------------------
            # (B) SAVE DATA OUTPUT AS CSV
            # -------------------------------------------------------------------------
            rows_grid, cols_grid = C_fast_current.shape
            lat_list = []
            lon_list = []
            landuse_list = []
            C_fast_list = []
            C_slow_list = []
            erosion_fast_list = []
            erosion_slow_list = []
            deposition_fast_list = []
            deposition_slow_list = []
            vegetation_fast_list = []
            vegetation_slow_list = []
            reaction_fast_list = []
            reaction_slow_list = []
            E_t_ha_list = []

            # Use constant p_fast_grid for flux calculations
            erosion_fast = -SOC_loss_g_kg_month * p_fast_grid_const
            erosion_slow = -SOC_loss_g_kg_month * (1 - p_fast_grid_const)

            dep_conc = (D_soc * 1000.0) / M_soil
            deposition_fast = dep_conc * p_fast_grid_const
            deposition_slow = dep_conc * (1 - p_fast_grid_const)

            vegetation_fast = V * p_fast_grid_const
            vegetation_slow = V * (1 - p_fast_grid_const)

            reaction_fast = -K_fast * C_fast_current
            reaction_slow = -K_slow * C_slow_current

            for i in range(rows_grid):
                for j in range(cols_grid):
                    cell_lon = grid_x[j]
                    cell_lat = grid_y[i]
                    lat_list.append(cell_lat)
                    lon_list.append(cell_lon)
                    landuse_list.append(str(LANDUSE[i, j]))
                    C_fast_list.append(C_fast_current[i, j])
                    C_slow_list.append(C_slow_current[i, j])
                    erosion_fast_list.append(erosion_fast[i, j])
                    erosion_slow_list.append(erosion_slow[i, j])
                    deposition_fast_list.append(deposition_fast[i, j])
                    deposition_slow_list.append(deposition_slow[i, j])
                    vegetation_fast_list.append(vegetation_fast[i, j])
                    vegetation_slow_list.append(vegetation_slow[i, j])
                    reaction_fast_list.append(reaction_fast[i, j])
                    reaction_slow_list.append(reaction_slow[i, j])
                    E_t_ha_list.append(E_t_ha_month[i, j])

            df_out = pd.DataFrame({
                'LAT': lat_list,
                'LON': lon_list,
                'Landuse': landuse_list,
                'C_fast': C_fast_list,
                'C_slow': C_slow_list,
                'Erosion_fast': erosion_fast_list,
                'Erosion_slow': erosion_slow_list,
                'Deposition_fast': deposition_fast_list,
                'Deposition_slow': deposition_slow_list,
                'Vegetation_fast': vegetation_fast_list,
                'Vegetation_slow': vegetation_slow_list,
                'Reaction_fast': reaction_fast_list,
                'Reaction_slow': reaction_slow_list,
                'E_t_ha_month': E_t_ha_list
            })

            filename_csv = f"Future_SOC_terms_{year}_{month_idx+1:02d}_timestep_{global_timestep}.csv"
            df_out.to_csv(os.path.join(OUTPUT_DIR / "Data", filename_csv), index=False)
            print(f"Saved CSV output for Year {year}, Month {month_idx+1} as {filename_csv}")

print("Future simulation (2025-2100) complete. Final SOC distribution is in C_fast_current + C_slow_current.")
