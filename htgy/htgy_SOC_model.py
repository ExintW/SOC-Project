import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import netCDF4 as nc

# =============================================================================
# 1) CSV READING & GRID SETUP
# =============================================================================

# Paths to CSV files containing spatial and dam information
region_csv_path = r"D:\EcoSci\Dr.Shi\Data\resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
dam_csv_path = r"D:\EcoSci\Dr.Shi\Data\htgy_Dam_with_matched_points.csv"
proportion_csv_path = r"D:\EcoSci\Dr.Shi\Data\Fast_Slow_SOC_Proportion.csv"

# Read the CSV files using pandas
df = pd.read_csv(region_csv_path, encoding='utf-8-sig')
df_dam = pd.read_csv(dam_csv_path, encoding='utf-8-sig')
df_prop = pd.read_csv(proportion_csv_path, encoding='utf-8-sig')

# =============================================================================
# DAM DATA: Convert columns to numeric and compute remaining capacity.
# Here, we assume that the dam CSV contains columns "total_stor" and "deposition"
# (both expressed in units of 10,000 m³), and the remaining capacity is:
#     capacity_remained = total_stor - deposition
# =============================================================================
df_dam["total_stor"] = pd.to_numeric(df_dam["total_stor"], errors='coerce')
df_dam["deposition"] = pd.to_numeric(df_dam["deposition"], errors='coerce')
df_dam["capacity_remained"] = df_dam["total_stor"] - df_dam["deposition"]

# =============================================================================
# Define column names from the region CSV.
# These columns include spatial coordinates, SOC concentration, DEM, land use,
# and the decay rates for fast and slow SOC pools.
# =============================================================================
lon_col, lat_col = "LON", "LAT"
soc_col = "ORGA"  # Initial SOC concentration in g/kg
dem_col = "htgy_DEM"
landuse_col = "LANDUSE"
region_col = "Region"
slope_col = "SLOPE"
k1_col = "SOC_k1_fast_pool (1/month)"
k2_col = "SOC_k2_slow_pool (1/month)"


def create_grid(data, col_name):
    """
    Create a 2D grid from CSV data using pivot.
    Rows = latitude (descending), columns = longitude (ascending).
    """
    return data.pivot(index=lat_col, columns=lon_col, values=col_name) \
        .sort_index(ascending=False).values


grid_x = np.sort(df[lon_col].unique())
grid_y = np.sort(df[lat_col].unique())[::-1]  # descending order for latitude

# Create 2D arrays from the CSV for each variable.
C = create_grid(df, soc_col)  # Initial SOC (g/kg)
DEM = create_grid(df, dem_col)  # Digital Elevation Model (elevation)
SAND = create_grid(df, "SAND")
SILT = create_grid(df, "SILT")
CLAY = create_grid(df, "CLAY")
LANDUSE = create_grid(df, landuse_col)
REGION = create_grid(df, region_col)
SLOPE = create_grid(df, slope_col)
K_fast = create_grid(df, k1_col)  # Fast pool decay rate (1/month)
K_slow = create_grid(df, k2_col)  # Slow pool decay rate (1/month)


# =============================================================================
# 2) PARTITION SOC INTO FAST & SLOW POOLS
# =============================================================================
def allocate_fast_slow_soc(C, LANDUSE, proportion_df):
    """
    Split total SOC (C) into fast and slow pools using land-use based proportions.
    The proportions are stored in df_prop with columns "Type", "Fast SOC(%)", and "Slow SOC(%)".
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
    p_fast_grid = np.zeros((rows, cols))  # Fraction of SOC in fast pool
    for i in range(rows):
        for j in range(cols):
            land_type = LANDUSE[i, j]
            # Default to all SOC in slow pool if land type not found
            props = prop_dict.get(land_type, {'fast': 0, 'slow': 1})
            C_fast[i, j] = C[i, j] * props['fast']
            C_slow[i, j] = C[i, j] * props['slow']
            p_fast_grid[i, j] = props['fast']
    return C_fast, C_slow, p_fast_grid


# Partition SOC into fast and slow pools.
C_fast, C_slow, p_fast_grid = allocate_fast_slow_soc(C, LANDUSE, df_prop)

# =============================================================================
# 3) DAM DATA PROCESSING
# =============================================================================
# Convert dam capacity from 10,000 m³ to tons.
# Bulk density is assumed to be 1.3 t/m³.
BULK_DENSITY = 1.3  # tons per cubic meter


def find_nearest_index(array, value):
    """Find the index of the array element closest to the given value."""
    return (np.abs(array - value)).argmin()


dam_positions = []
for _, row in df_dam.iterrows():
    i_idx = find_nearest_index(grid_y, row["y"])
    j_idx = find_nearest_index(grid_x, row["x"])
    capacity_10000_m3 = row["capacity_remained"]  # Capacity in units of 10,000 m³
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
    Calculate the monthly rainfall erosivity factor (R factor).
    Empirical formula: R = 1.735 * monthly_rain_mm.
    (Adjust the coefficient as necessary.)
    """
    return 1.735 * rain_month_mm


def calculate_ls_factor(slope, slope_length=1000):
    """
    Calculate the slope length and steepness factor (LS factor).
    The function uses an empirical formula with slope in degrees.
    """
    slope_rad = np.deg2rad(slope)
    return ((slope_length / 22.13) ** 0.4) * ((np.sin(slope_rad) / 0.0896) ** 1.3)


def calculate_c_factor(lai):
    """
    Calculate the cover-management factor (C factor) based on LAI.
    Here, an exponential relationship is assumed.
    """
    return np.exp(-2.5 * lai)


def calculate_p_factor(landuse):
    """
    Return the support practice factor (P factor) based on land use.
    """
    p_values = {
        "Sloping cropland": 0.6,
        "Forestland": 0.4,
        "Grassland": 0.5,
        "Terrace": 0.1,
        "Dam field": 0.05
    }
    return p_values.get(landuse, 1.0)


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
    Convert soil loss E (in t/ha/month) to SOC loss in g/kg/month.
    1 t/ha = 100 g/m².
    """
    E_g_m2_month = E_t_ha_month * 100.0  # Convert from t/ha to g/m²
    soc_loss_g_m2_month = E_g_m2_month * (ORGA_g_per_kg / 1000.0) * bulk_density
    soc_loss_g_kg_month = soc_loss_g_m2_month / bulk_density
    return soc_loss_g_kg_month


# =============================================================================
# 6) ROUTE SOIL AND SOC FROM HIGHER CELLS
# =============================================================================
def distribute_soil_and_soc_with_dams(E_tcell, S, DEM, dam_positions, grid_x, grid_y):
    """
    Routes soil and its associated SOC that comes from cells higher up in the landscape.

    Key points:
      - Only the soil/SOC arriving from upstream (inflow) is deposited in a cell.
      - The cell's own erosion (E_tcell and S) is not deposited locally; it is routed downstream.
      - If a cell contains a dam, all upstream inflow is retained until the dam is full.
        Once full, any excess is routed downstream.

    Parameters:
      E_tcell : 2D array (t/cell/month)
          Soil loss generated locally in each cell (from RUSLE).
      S : 2D array (kg/cell/month)
          SOC mass eroded from each cell.
      DEM : 2D array
          Elevation for each cell.
      dam_positions : list of dict
          Each dictionary contains 'i', 'j', and 'capacity_remained_tons' (in tons).
      grid_x, grid_y : 1D arrays
          Coordinates of the cell centers.

    Returns:
      D_soil : 2D array (t/cell/month)
          Soil mass deposited in each cell from upstream (inflow only).
      D_soc : 2D array (kg/cell/month)
          SOC mass deposited in each cell from upstream.
      inflow_soil, inflow_soc : 2D arrays
          The remaining soil and SOC mass that continues to flow downstream.
    """
    rows, cols = DEM.shape

    # Create a dam capacity map for quick lookup.
    dam_capacity_map = {(dam['i'], dam['j']): dam['capacity_remained_tons'] for dam in dam_positions}

    # Initialize arrays to store inflow and deposition.
    inflow_soil = np.zeros_like(E_tcell, dtype=float)
    inflow_soc = np.zeros_like(E_tcell, dtype=float)
    D_soil = np.zeros_like(E_tcell, dtype=float)
    D_soc = np.zeros_like(E_tcell, dtype=float)

    # Build a list of all cells and sort them from highest to lowest elevation.
    all_cells = [(i, j) for i in range(rows) for j in range(cols)]
    all_cells.sort(key=lambda c: DEM[c[0], c[1]], reverse=True)

    # Process each cell in descending order.
    for (i, j) in all_cells:
        # The deposition available in the cell is the inflow from upstream.
        deposition_soil = inflow_soil[i, j]
        deposition_soc = inflow_soc[i, j]

        # If the cell contains a dam, retain all incoming deposition up to capacity.
        if (i, j) in dam_capacity_map:
            cap = dam_capacity_map[(i, j)]
            if deposition_soil <= cap:
                # All incoming soil/SOC is stored.
                D_soil[i, j] = deposition_soil
                D_soc[i, j] = deposition_soc
                dam_capacity_map[(i, j)] = cap - deposition_soil
                excess_soil = 0
                excess_soc = 0
            else:
                # Only store up to the dam's capacity.
                D_soil[i, j] = cap
                deposited_soc = (cap / deposition_soil) * deposition_soc if deposition_soil > 0 else 0
                D_soc[i, j] = deposited_soc
                dam_capacity_map[(i, j)] = 0  # dam becomes full
                excess_soil = deposition_soil - cap
                excess_soc = deposition_soc - deposited_soc
            current_inflow_soil = excess_soil
            current_inflow_soc = excess_soc
        else:
            # In cells without a dam, all inflow is deposited.
            D_soil[i, j] = deposition_soil
            D_soc[i, j] = deposition_soc
            current_inflow_soil = deposition_soil
            current_inflow_soc = deposition_soc

        # The cell's own erosion is not deposited locally.
        # It is immediately routed downstream.
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
                        # Calculate a weight based on the slope difference.
                        slope_diff = (DEM[i, j] - DEM[ni, nj]) / np.hypot(di, dj)
                        neighbors.append(((ni, nj), slope_diff))
        total_slope = sum(s for _, s in neighbors)
        if total_slope > 0:
            # Distribute the locally eroded soil and SOC to lower neighbors.
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
    Compute vegetation input based on LAI.
    The empirical formula is provided (adjust as needed).
    """
    return 0.00008128 * (LAI ** 7.33382537)


def soc_dynamic_model(C_fast, C_slow,
                      soc_loss_g_kg_month, D_soil, D_soc, V,
                      K_fast, K_slow, p_fast_grid, dt, M_soil=1500.0):
    """
    Update the SOC pools (g/kg) for one month.

    Parameters:
      C_fast, C_slow: 2D arrays of current SOC concentrations (g/kg)
      soc_loss_g_kg_month: SOC loss due to erosion from the cell (g/kg/month)
      D_soil: Deposited soil mass from upstream (t/cell/month)
      D_soc: Deposited SOC mass from upstream (kg/cell/month)
      V: Vegetation input (g/kg/month)
      K_fast, K_slow: First-order decay rates (1/month) for fast and slow pools
      p_fast_grid: Fraction of SOC in fast pool
      dt: Time step (in months)
      M_soil: Total mass of soil per cell (kg); used for conversion

    The deposition from upstream (D_soc, in kg) is converted to an increase in SOC concentration (g/kg)
    using the formula: deposition concentration = (D_soc * 1000) / M_soil.
    """
    # Erosion removes SOC from the pools.
    erosion_fast = -soc_loss_g_kg_month * p_fast_grid
    erosion_slow = -soc_loss_g_kg_month * (1 - p_fast_grid)

    # Convert deposited SOC mass (kg) to a concentration increment (g/kg).
    deposition_concentration = (D_soc * 1000.0) / M_soil  # kg -> g
    deposition_fast = deposition_concentration * p_fast_grid
    deposition_slow = deposition_concentration * (1 - p_fast_grid)

    # Vegetation input adds SOC.
    vegetation_fast = V * p_fast_grid
    vegetation_slow = V * (1 - p_fast_grid)

    # Decay (first-order loss).
    reaction_fast = -K_fast * C_fast
    reaction_slow = -K_slow * C_slow

    # Update the SOC pools.
    C_fast_new = np.maximum(C_fast + (erosion_fast + deposition_fast + vegetation_fast + reaction_fast) * dt, 0)
    C_slow_new = np.maximum(C_slow + (erosion_slow + deposition_slow + vegetation_slow + reaction_slow) * dt, 0)

    return C_fast_new, C_slow_new


# =============================================================================
# 8) HELPER: REGRID ERA5 POINT DATA TO 2D GRID
# =============================================================================
def create_grid_from_points(lon_points, lat_points, values, grid_x, grid_y):
    """
    Regrid 1D ERA5 point data to a 2D grid.

    For each ERA5 point, find the nearest grid cell (using grid_x and grid_y as cell centers).
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
output_folder = r"D:\EcoSci\Dr.Shi\Figure_output"
os.makedirs(output_folder, exist_ok=True)

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
plt.savefig(os.path.join(output_folder, "SOC_initial.png"))
plt.close(fig)

# =============================================================================
# 10) MAIN SIMULATION LOOP (MONTHLY, 2007-2025)
# =============================================================================
CELL_AREA_HA = 100.0  # 1 km² = 100 ha
start_year = 2007
end_year = 2025
global_timestep = 0

# M_soil: total mass of soil per cell (kg).
# Adjust for your study area. This is used to convert D_soc (kg) to g/kg increments.
M_soil = 1e8  # placeholder value

C_fast_current = C_fast.copy()
C_slow_current = C_slow.copy()

# New directory to save monthly CSV outputs
output_data_dir = r"D:\EcoSci\Dr.Shi\Data_output"
os.makedirs(output_data_dir, exist_ok=True)

for year in range(start_year, end_year + 1):
    nc_file = rf"D:\EcoSci\Dr.Shi\Data\ERA5\Download\ERA5_Data_Monthly_Resampled\resampled_{year}.nc"
    if not os.path.exists(nc_file):
        print(f"NetCDF file not found for year {year}: {nc_file}")
        continue

    with nc.Dataset(nc_file) as ds:
        valid_time = ds.variables['valid_time'][:]  # expecting 12 timesteps (months)
        n_time = len(valid_time)

        # 1D arrays for ERA5 lat/lon
        lon_nc = ds.variables['longitude'][:]
        lat_nc = ds.variables['latitude'][:]

        # LAI (lai_lv) and precipitation (tp)
        lai_data = ds.variables['lai_lv'][:]  # shape: (12, n_points)
        tp_data = ds.variables['tp'][:]  # shape: (12, n_points), in meters

        for month_idx in range(n_time):
            # Regrid LAI
            lai_1d = lai_data[month_idx, :]
            LAI_2D = create_grid_from_points(lon_nc, lat_nc, lai_1d, grid_x, grid_y)

            # Regrid precipitation (tp), convert from meters to mm
            tp_1d = tp_data[month_idx, :]
            tp_1d_mm = tp_1d * 1000.0
            RAIN_2D = create_grid_from_points(lon_nc, lat_nc, tp_1d_mm, grid_x, grid_y)

            # RUSLE factors
            R_month = calculate_r_factor_monthly(RAIN_2D)
            C_factor_2D = calculate_c_factor(LAI_2D)

            # Soil loss in t/ha/month
            E_t_ha_month = R_month * K_factor * LS_factor * C_factor_2D * P_factor
            # Convert to t/cell/month
            E_tcell_month = E_t_ha_month * CELL_AREA_HA

            # SOC mass eroded from each cell (kg/cell/month)
            S = E_tcell_month * (C_fast_current + C_slow_current)

            # SOC loss in g/kg/month (erosion from the cell)
            SOC_loss_g_kg_month = convert_soil_loss_to_soc_loss_monthly(
                E_t_ha_month, (C_fast_current + C_slow_current)
            )

            # Route soil and SOC from higher cells
            D_soil, D_soc, inflow_soil, inflow_soc = distribute_soil_and_soc_with_dams(
                E_tcell_month, S, DEM, dam_positions, grid_x, grid_y
            )

            # Vegetation input (g/kg/month) from LAI
            V = vegetation_input(LAI_2D)

            # Update SOC pools (fast & slow)
            C_fast_current, C_slow_current = soc_dynamic_model(
                C_fast_current, C_slow_current,
                SOC_loss_g_kg_month, D_soil, D_soc, V,
                K_fast, K_slow, p_fast_grid,
                dt=1,  # 1 month
                M_soil=M_soil
            )

            global_timestep += 1
            print(f"Completed simulation for Year {year}, Month {month_idx + 1}")

            # === (A) SAVE FIGURE ===
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
            plt.savefig(os.path.join(output_folder, filename_fig))
            plt.close(fig)

            # === (B) SAVE DATA AS CSV ===
            # Flatten the C_fast_current & C_slow_current arrays into a DataFrame
            rows, cols = C_fast_current.shape
            lat_list = []
            lon_list = []
            cfast_list = []
            cslow_list = []

            for i in range(rows):
                for j in range(cols):
                    lat_list.append(grid_y[i])
                    lon_list.append(grid_x[j])
                    cfast_list.append(C_fast_current[i, j])
                    cslow_list.append(C_slow_current[i, j])

            df_out = pd.DataFrame({
                'LAT': lat_list,
                'LON': lon_list,
                'C_fast': cfast_list,
                'C_slow': cslow_list,
                'C_total': [f + s for f, s in zip(cfast_list, cslow_list)]
            })

            # Construct a filename with year, month, and timestep info
            filename_csv = f"SOC_{year}_{month_idx + 1:02d}_timestep_{global_timestep}.csv"
            df_out.to_csv(os.path.join(output_data_dir, filename_csv), index=False)

print("Simulation complete. Final SOC distribution is in C_fast_current + C_slow_current.")
