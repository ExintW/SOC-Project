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
# 1) CSV READING & GRID SETUP
# =============================================================================
# Define file paths for the region CSV, dam CSV, and SOC proportion CSV.
region_csv_path = DATA_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
dam_csv_path = DATA_DIR / "htgy_Dam_with_matched_points.csv"
proportion_csv_path = DATA_DIR / "Fast_Slow_SOC_Proportion.csv"

# Read the CSV files into DataFrames.
df = pd.read_csv(region_csv_path, encoding='utf-8-sig')
df_dam = pd.read_csv(dam_csv_path, encoding='utf-8-sig')
df_prop = pd.read_csv(proportion_csv_path, encoding='utf-8-sig')

# -----------------------------------------------------------------------------
# DAM DATA PROCESSING:
# Convert "total_stor" and "deposition" columns to numeric, then compute the remaining
# capacity as: capacity_remained = total_stor - deposition.
# Units are assumed to be in 10,000 m³.
# -----------------------------------------------------------------------------
df_dam["total_stor"] = pd.to_numeric(df_dam["total_stor"], errors='coerce')
df_dam["deposition"] = pd.to_numeric(df_dam["deposition"], errors='coerce')
df_dam["capacity_remained"] = df_dam["total_stor"] - df_dam["deposition"]

# -----------------------------------------------------------------------------
# Define column names from the region CSV.
# These include spatial coordinates, SOC concentration, DEM, land use, and decay rates.
# -----------------------------------------------------------------------------
lon_col, lat_col = "LON", "LAT"
soc_col = "ORGA"  # Initial SOC concentration (g/kg)
dem_col = "htgy_DEM"  # Digital Elevation Model (elevation)
landuse_col = "LANDUSE"  # Land use
region_col = "Region"  # (Available but not used later)
slope_col = "SLOPE"  # Slope values
k1_col = "SOC_k1_fast_pool (1/month)"  # Fast pool decay rate
k2_col = "SOC_k2_slow_pool (1/month)"  # Slow pool decay rate



# -----------------------------------------------------------------------------
# Function: create_grid
# Creates a 2D grid using a pivot operation:
#   - Rows are unique latitudes (sorted in descending order)
#   - Columns are unique longitudes (sorted in ascending order)
# -----------------------------------------------------------------------------
def create_grid(data, col_name):
    return data.pivot(index=lat_col, columns=lon_col, values=col_name) \
        .sort_index(ascending=False).values


# -----------------------------------------------------------------------------
# Extract unique grid coordinates from the region CSV.
# -----------------------------------------------------------------------------
grid_x = np.sort(df[lon_col].unique())
grid_y = np.sort(df[lat_col].unique())[::-1]  # Descending order for latitude

# -----------------------------------------------------------------------------
# Create 2D arrays for each variable using the CSV.
# -----------------------------------------------------------------------------
C = create_grid(df, soc_col)  # Initial SOC (g/kg)
# Clip initial SOC values: set any value above 13.8 to 13.8.
C = np.clip(C, None, 13.8)
DEM = create_grid(df, dem_col)  # Elevation
SAND = create_grid(df, "SAND")
SILT = create_grid(df, "SILT")
CLAY = create_grid(df, "CLAY")
LANDUSE = create_grid(df, landuse_col)  # Land use grid (will be saved to output)
REGION = create_grid(df, region_col)
SLOPE = create_grid(df, slope_col)
K_fast = create_grid(df, k1_col)  # Fast pool decay rate (1/month)
K_slow = create_grid(df, k2_col)  # Slow pool decay rate (1/month)
# After pivoting to create grids, fill missing DEM and SLOPE values:
DEM = np.nan_to_num(DEM, nan=np.nanmean(DEM))
SLOPE = np.nan_to_num(SLOPE, nan=np.nanmean(SLOPE))
K_fast = np.nan_to_num(K_fast, nan=np.nanmean(K_fast))
K_slow = np.nan_to_num(K_slow, nan=np.nanmean(K_slow))


# =============================================================================
# 2) PARTITION SOC INTO FAST & SLOW POOLS
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
    C_fast = np.zeros((rows, cols))
    C_slow = np.zeros((rows, cols))
    p_fast_grid = np.zeros((rows, cols))  # Fraction of SOC in the fast pool.
    for i in range(rows):
        for j in range(cols):
            land_type = LANDUSE[i, j]
            # If land type is not found, assume all SOC goes to the slow pool.
            props = prop_dict.get(land_type, {'fast': 0, 'slow': 1})
            C_fast[i, j] = C[i, j] * props['fast']
            C_slow[i, j] = C[i, j] * props['slow']
            p_fast_grid[i, j] = props['fast']
    return C_fast, C_slow, p_fast_grid


C_fast, C_slow, p_fast_grid = allocate_fast_slow_soc(C, LANDUSE, df_prop)

# =============================================================================
# 3) DAM DATA PROCESSING
# =============================================================================
# Convert dam capacity from 10,000 m³ to tons.
# We assume a bulk density of 1300 t/m³.
BULK_DENSITY = 1300  # Tons per cubic meter


def find_nearest_index(array, value):
    """Return the index of the element in 'array' closest to 'value'."""
    return (np.abs(array - value)).argmin()


dam_positions = []
for _, row in df_dam.iterrows():
    i_idx = find_nearest_index(grid_y, row["y"])
    j_idx = find_nearest_index(grid_x, row["x"])
    capacity_10000_m3 = row["capacity_remained"]  # In units of 10,000 m³.
    # Convert to tons: multiply by 10,000 (to get m³) then by BULK_DENSITY.
    capacity_tons = capacity_10000_m3 * 10_000 * BULK_DENSITY
    dam_positions.append({
        'i': i_idx,
        'j': j_idx,
        'capacity_remained_tons': capacity_tons
    })


# =============================================================================
# 4) RUSLE COMPONENTS (MONTHLY)
# =============================================================================
def calculate_r_factor_monthly(rain_month_mm):
    """
    Compute the monthly rainfall erosivity factor (R factor).

    For the Loess Plateau, studies (e.g., Zhao et al. 2012) suggest a coefficient
    about 4 times higher than the standard value, so:

        R = 6.94 * rain_month_mm

    This adjustment yields soil loss values closer to observed rates (~1000 t/km²/year).
    """
    return 6.94 * rain_month_mm


def calculate_ls_factor(slope, slope_length=1000):
    """
    Compute the LS factor (slope length and steepness factor) from slope.
    Slope is provided in degrees.
    Uses the standard RUSLE formulation.
    """
    slope_rad = np.deg2rad(slope)
    return ((slope_length / 22.13) ** 0.4) * ((np.sin(slope_rad) / 0.0896) ** 1.3)


def calculate_c_factor(lai):
    """
    Compute the C factor from LAI using an exponential decay.

    For the Loess Plateau, using a lower exponent (1.7 instead of 2.5) yields higher
    soil loss values that better match observed data.

        C = exp(-1.7 * LAI)

    Sources: Zhou (2008); Wei & Liu (2016).
    """
    return np.exp(-1.7 * lai)


def calculate_p_factor(landuse):
    """
    Return the support practice factor (P factor) based on land use.

    For the Loess Plateau, recommended values are:
      - "sloping cropland": 0.4  (conservation measures)
      - "forestland": 1.0        (no additional support practices)
      - "grassland": 1.0
      - "not used": 1.0
      - "terrace": 0.1          (very effective conservation)
      - "dam field": 0.05       (extremely low due to dam sediment trapping)

    The input is converted to a string and lowercased.
    Sources: Gao et al. (2016); Liu et al. (2001).
    """
    p_values = {
        "sloping cropland": 0.4,
        "forestland": 1.0,
        "grassland": 1.0,
        "not used": 1.0,
        "terrace": 0.1,
        "dam field": 0.05
    }
    return p_values.get(str(landuse).lower(), 1.0)


# Set constant K factor.
K_factor = np.full_like(C, 0.03)
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
    Note: 1 t/ha = 100 g/m².
    """
    E_g_m2_month = E_t_ha_month * 100.0  # Convert from t/ha to g/m².
    soc_loss_g_m2_month = E_g_m2_month * (ORGA_g_per_kg / 1000.0) * bulk_density
    soc_loss_g_kg_month = soc_loss_g_m2_month / bulk_density
    return soc_loss_g_kg_month


# =============================================================================
# 6) ROUTE SOIL AND SOC FROM HIGHER CELLS
# =============================================================================
def distribute_soil_and_soc_with_dams(E_tcell, S, DEM, dam_positions, grid_x, grid_y):
    """
    Route soil and its associated SOC arriving from upstream cells.

    Key points:
      - Only the soil/SOC arriving from upstream (inflow) is deposited in a cell.
      - The cell's own erosion is not deposited locally; it is immediately routed downstream.
      - For cells with a dam, upstream inflow is retained up to the dam's capacity; any excess is routed downstream.

    Parameters:
      E_tcell : 2D array (t/cell/month)
          Soil loss generated locally (from RUSLE).
      S : 2D array (kg/cell/month)
          SOC mass eroded locally.
      DEM : 2D array
          Elevation for each cell.
      dam_positions : list of dicts with keys 'i', 'j', 'capacity_remained_tons'.
      grid_x, grid_y : 1D arrays of cell center coordinates.

    Returns:
      D_soil : 2D array (t/cell/month) of soil deposited (from upstream inflow).
      D_soc : 2D array (kg/cell/month) of SOC deposited (from upstream inflow).
      inflow_soil, inflow_soc : 2D arrays of soil and SOC still flowing downstream.
    """
    rows, cols = DEM.shape
    # Build a dam capacity map.
    dam_capacity_map = {(dam['i'], dam['j']): dam['capacity_remained_tons'] for dam in dam_positions}
    # Initialize inflow and deposition arrays.
    inflow_soil = np.zeros_like(E_tcell, dtype=float)
    inflow_soc = np.zeros_like(E_tcell, dtype=float)
    D_soil = np.zeros_like(E_tcell, dtype=float)
    D_soc = np.zeros_like(E_tcell, dtype=float)
    # Create a list of all cells and sort them from highest to lowest elevation.
    all_cells = [(i, j) for i in range(rows) for j in range(cols)]
    all_cells.sort(key=lambda c: DEM[c[0], c[1]], reverse=True)

    for (i, j) in all_cells:
        # The deposition available in the cell is the upstream inflow.
        deposition_soil = inflow_soil[i, j]
        deposition_soc = inflow_soc[i, j]
        # Ensure inflow values are non-negative.
        deposition_soil = max(deposition_soil, 0)
        deposition_soc = max(deposition_soc, 0)

        # If the cell contains a dam, retain inflow up to the dam's capacity.
        if (i, j) in dam_capacity_map:
            cap = dam_capacity_map[(i, j)]
            if deposition_soil <= cap:
                D_soil[i, j] = deposition_soil
                D_soc[i, j] = deposition_soc
                dam_capacity_map[(i, j)] = max(0, cap - deposition_soil)
                excess_soil = 0
                excess_soc = 0
            else:
                D_soil[i, j] = cap
                deposited_soc = (cap / deposition_soil) * deposition_soc if deposition_soil > 0 else 0
                # Clamp deposited_soc to non-negative.
                deposited_soc = max(0, deposited_soc)
                D_soc[i, j] = deposited_soc
                dam_capacity_map[(i, j)] = 0  # Dam is now full.
                excess_soil = deposition_soil - cap
                excess_soc = deposition_soc - deposited_soc
            current_inflow_soil = excess_soil
            current_inflow_soc = excess_soc
        else:
            D_soil[i, j] = deposition_soil
            D_soc[i, j] = deposition_soc
            current_inflow_soil = deposition_soil
            current_inflow_soc = deposition_soc

        # The cell's own erosion (source) is not deposited locally.
        source_soil = E_tcell[i, j]
        source_soc = S[i, j]

        # Identify lower neighbors (cells with lower elevation).
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
            # Distribute the cell's own erosion to lower neighbors.
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
    Compute vegetation input based on LAI using an empirical formula.
    """
    return 0.00008128 * (LAI ** 7.33382537)


def soc_dynamic_model(C_fast, C_slow,
                      soc_loss_g_kg_month, D_soil, D_soc, V,
                      K_fast, K_slow, p_fast_grid, dt, M_soil):
    """
    Update the SOC pools (g/kg) for one month.

    Deposited SOC (D_soc, in kg) is converted to a concentration increment (g/kg)
    using: deposition_concentration = (D_soc * 1000) / M_soil.

    Parameters:
      C_fast, C_slow : Current SOC concentrations (g/kg).
      soc_loss_g_kg_month : SOC loss from erosion (g/kg/month).
      D_soil : Deposited soil from upstream (t/cell/month).
      D_soc : Deposited SOC from upstream (kg/cell/month).
      V : Vegetation input (g/kg/month).
      K_fast, K_slow : Decay rates (1/month) for fast and slow pools.
      p_fast_grid : Fraction of SOC in the fast pool.
      dt : Timestep (months).
      M_soil : Total soil mass per cell (kg).
    """
    # Compute erosion losses.
    erosion_fast = -soc_loss_g_kg_month * p_fast_grid
    erosion_slow = -soc_loss_g_kg_month * (1 - p_fast_grid)

    # Convert deposited SOC (kg) to a concentration increment (g/kg).
    deposition_concentration = (D_soc * 1000.0) / M_soil
    deposition_fast = deposition_concentration * p_fast_grid
    deposition_slow = deposition_concentration * (1 - p_fast_grid)

    # Vegetation input adds SOC.
    vegetation_fast = V * p_fast_grid
    vegetation_slow = V * (1 - p_fast_grid)

    # Reaction (decay) losses.
    reaction_fast = -K_fast * C_fast
    reaction_slow = -K_slow * C_slow

    # Update SOC pools.
    C_fast_new = np.maximum(C_fast + (erosion_fast + deposition_fast + vegetation_fast + reaction_fast) * dt, 0)
    C_slow_new = np.maximum(C_slow + (erosion_slow + deposition_slow + vegetation_slow + reaction_slow) * dt, 0)

    return C_fast_new, C_slow_new

# =============================================================================
# 8) HELPER: REGRID ERA5 POINT DATA TO 2D GRID
# =============================================================================
def create_grid_from_points(lon_points, lat_points, values, grid_x, grid_y):
    """
    Regrid 1D ERA5 point data to a 2D grid.
    For each ERA5 point, find the nearest grid cell using grid_x and grid_y as cell centers.
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

# Plot and save the initial SOC distribution.
fig, ax = plt.subplots()
cax = ax.imshow(C_fast + C_slow, cmap="viridis",
                extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
cbar = fig.colorbar(cax, label="SOC (g/kg)")
ax.set_title("Initial SOC Distribution (t = 0)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
ax.ticklabel_format(style='plain', axis='x')
plt.savefig(os.path.join(OUTPUT_DIR, "SOC_initial.png"))
plt.close(fig)

# =============================================================================
# 10) MAIN SIMULATION LOOP (MONTHLY, 2007-2025)
# =============================================================================
CELL_AREA_HA = 100.0  # 1 km² = 100 ha
start_year = 2007
end_year = 2025
global_timestep = 0

# M_soil: Total soil mass per cell (kg).
# =============================================================================
# Calculate the effective soil mass per cell (M_soil)
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

M_soil = 1.0e8  # Total soil mass per cell (kg) based on 0.077 m depth and 1300 kg/m³

# (The remainder of your code uses M_soil to convert deposited SOC mass (kg/cell/month)
# into a concentration increment (g/kg/month) via the formula:
#    deposition_concentration (g/kg) = (D_soc * 1000) / M_soil
# )

# Initialize current SOC pools with the initial values.
C_fast_current = C_fast.copy()
C_slow_current = C_slow.copy()

# Create a directory to save monthly CSV outputs.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Main simulation loop: Iterate over each year and month.
# -----------------------------------------------------------------------------
for year in range(start_year, end_year + 1):
    nc_file = DATA_DIR / "ERA5" / f"resampled_{year}.nc"
    if not os.path.exists(nc_file):
        print(f"NetCDF file not found for year {year}: {nc_file}")
        continue

    with nc.Dataset(nc_file) as ds:
        valid_time = ds.variables['valid_time'][:]  # Expecting 12 months.
        n_time = len(valid_time)

        # Get ERA5 coordinate arrays.
        lon_nc = ds.variables['longitude'][:]
        lat_nc = ds.variables['latitude'][:]

        # Read LAI (lai_lv) and precipitation (tp) data.
        lai_data = ds.variables['lai_lv'][:]  # Shape: (12, n_points)
        tp_data = ds.variables['tp'][:]  # Shape: (12, n_points), in meters

        for month_idx in range(n_time):
            # -------------------------------------------------------------------------
            # Regrid LAI data to the model grid.
            # -------------------------------------------------------------------------
            lai_1d = lai_data[month_idx, :]
            LAI_2D = create_grid_from_points(lon_nc, lat_nc, lai_1d, grid_x, grid_y)
            LAI_2D = np.nan_to_num(LAI_2D, nan=np.nanmean(LAI_2D))

            # -------------------------------------------------------------------------
            # Regrid precipitation data and convert from meters to mm.
            # -------------------------------------------------------------------------
            tp_1d = tp_data[month_idx, :]
            tp_1d_mm = tp_1d * 1000.0
            RAIN_2D = create_grid_from_points(lon_nc, lat_nc, tp_1d_mm, grid_x, grid_y)
            RAIN_2D = np.nan_to_num(RAIN_2D, nan=np.nanmean(RAIN_2D))

            # -------------------------------------------------------------------------
            # Compute RUSLE factors based on regridded data.
            # -------------------------------------------------------------------------
            R_month = calculate_r_factor_monthly(RAIN_2D)
            C_factor_2D = calculate_c_factor(LAI_2D)

            # -------------------------------------------------------------------------
            # Calculate soil loss (E) in t/ha/month, then convert to t/cell/month.
            # -------------------------------------------------------------------------
            E_t_ha_month = R_month * K_factor * LS_factor * C_factor_2D * P_factor
            E_tcell_month = E_t_ha_month * CELL_AREA_HA

            # -------------------------------------------------------------------------
            # Compute SOC mass eroded from each cell (kg/cell/month).
            # S = E_tcell_month * (C_fast_current + C_slow_current)
            # -------------------------------------------------------------------------
            S = E_tcell_month * (C_fast_current + C_slow_current)

            # -------------------------------------------------------------------------
            # Compute SOC loss (erosion) in g/kg/month.
            # -------------------------------------------------------------------------
            SOC_loss_g_kg_month = convert_soil_loss_to_soc_loss_monthly(
                E_t_ha_month, (C_fast_current + C_slow_current)
            )

            # -------------------------------------------------------------------------
            # Route soil and SOC from higher (upstream) cells.
            # Note: The cell's own erosion is not deposited locally.
            # -------------------------------------------------------------------------
            D_soil, D_soc, inflow_soil, inflow_soc = distribute_soil_and_soc_with_dams(
                E_tcell_month, S, DEM, dam_positions, grid_x, grid_y
            )

            # -------------------------------------------------------------------------
            # Compute vegetation input (g/kg/month) from LAI.
            # -------------------------------------------------------------------------
            V = vegetation_input(LAI_2D)

            # -------------------------------------------------------------------------
            # Compute individual terms of the SOC balance equation.
            # -------------------------------------------------------------------------
            erosion_fast = -SOC_loss_g_kg_month * p_fast_grid
            erosion_slow = -SOC_loss_g_kg_month * (1 - p_fast_grid)

            # Deposited SOC is converted to a concentration increment (g/kg)
            deposition_concentration = (D_soc * 1000.0) / M_soil  # (kg -> g conversion)
            deposition_fast = deposition_concentration * p_fast_grid
            deposition_slow = deposition_concentration * (1 - p_fast_grid)

            vegetation_fast = V * p_fast_grid
            vegetation_slow = V * (1 - p_fast_grid)

            reaction_fast = -K_fast * C_fast_current
            reaction_slow = -K_slow * C_slow_current

            # -------------------------------------------------------------------------
            # Update SOC pools using the dynamic model.
            # -------------------------------------------------------------------------
            # In your main simulation loop, update the call to soc_dynamic_model as follows:
            C_fast_current, C_slow_current = soc_dynamic_model(
                C_fast_current, C_slow_current,
                SOC_loss_g_kg_month, D_soil, D_soc, V,
                K_fast, K_slow, p_fast_grid,
                dt=1,  # 1-month timestep.
                M_soil=M_soil
            )

            global_timestep += 1
            print(f"Completed simulation for Year {year}, Month {month_idx + 1}")

            # -------------------------------------------------------------------------
            # (A) SAVE FIGURE OUTPUT.
            # -------------------------------------------------------------------------
            fig, ax = plt.subplots()
            cax = ax.imshow(C_fast_current + C_slow_current, cmap="viridis",
                            extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
            cbar = fig.colorbar(cax, label="SOC (g/kg)")
            ax.set_title(f"SOC at Timestep {global_timestep} (Year {year}, Month {month_idx + 1})")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='x')
            filename_fig = f"SOC_{year}_{month_idx + 1:02d}_timestep_{global_timestep}.png"
            plt.savefig(os.path.join(OUTPUT_DIR / "Figure", filename_fig))
            plt.close(fig)

            # -------------------------------------------------------------------------
            # (B) SAVE DATA OUTPUT AS CSV.
            # Save each term in the SOC balance equation along with E_t_ha_month and landuse.
            # In this version, we include all grid cells (no spatial clipping).
            # -------------------------------------------------------------------------
            rows_grid, cols_grid = C_fast_current.shape
            lat_list = []
            lon_list = []
            landuse_list = []  # New list for landuse type
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

            for i in range(rows_grid):
                for j in range(cols_grid):
                    cell_lon = grid_x[j]
                    cell_lat = grid_y[i]
                    # Include all cells (no border clipping).
                    lat_list.append(cell_lat)
                    lon_list.append(cell_lon)
                    # Include the landuse type.
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

            filename_csv = f"SOC_terms_{year}_{month_idx + 1:02d}_timestep_{global_timestep}.csv"
            df_out.to_csv(os.path.join(OUTPUT_DIR / "Data", filename_csv), index=False)
            print(f"Saved CSV output for Year {year}, Month {month_idx + 1} as {filename_csv}")

print("Simulation complete. Final SOC distribution is in C_fast_current + C_slow_current.")