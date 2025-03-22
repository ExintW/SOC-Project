import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import netCDF4 as nc
import geopandas as gpd
from shapely.geometry import Point
from shapely import vectorized  # Not used now; we'll use np.vectorize instead.
from shapely.prepared import prep
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # Expects DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

# =============================================================================
# SET DESIRED CRS FOR THE MODEL
# =============================================================================
# Your main grid is in degrees, so we want everything in EPSG:4326.
desired_crs = "EPSG:4326"

# =============================================================================
# 0) SETUP: READ THE LOESS PLATEAU BORDER SHAPEFILE
# =============================================================================
# Read the Loess Plateau border shapefile and combine all features into one geometry.
loess_border_path = DATA_DIR / "Loess_Plateau_vector_border.shp"
loess_border = gpd.read_file(loess_border_path)
# Use union_all() to merge all features (recommended over the deprecated unary_union).
loess_border_geom = loess_border.union_all()
# Reproject the Loess Plateau border to the desired CRS.
loess_border = loess_border.to_crs(desired_crs)
loess_border_geom = loess_border.union_all()

# =============================================================================
# 1) CSV READING & GRID SETUP
# =============================================================================
# Define file paths for the region CSV, dam CSV, and SOC proportion CSV.
region_csv_path = PROCESSED_DIR / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
dam_csv_path = PROCESSED_DIR / "htgy_Dam_with_matched_points.csv"
proportion_csv_path = DATA_DIR / "Fast_Slow_SOC_Proportion.csv"

# Read the CSV files into DataFrames.
df = pd.read_csv(region_csv_path, encoding='utf-8-sig')
df_dam = pd.read_csv(dam_csv_path, encoding='utf-8-sig')
df_prop = pd.read_csv(proportion_csv_path, encoding='utf-8-sig')

# --- NEW: Ensure the 'year' column (dam construction year) is numeric ---
df_dam["year"] = pd.to_numeric(df_dam["year"], errors="coerce")
df_dam["total_stor"] = pd.to_numeric(df_dam["total_stor"], errors="coerce")
df_dam["deposition"] = pd.to_numeric(df_dam["deposition"], errors="coerce")
df_dam["capacity_remained"] = df_dam["total_stor"] - df_dam["deposition"]

# -----------------------------------------------------------------------------
# Define column names from the region CSV.
# These include spatial coordinates, SOC concentration, DEM, land use, and decay rates.
# -----------------------------------------------------------------------------
lon_col, lat_col = "LON", "LAT"
soc_col = "ORGA"           # Initial SOC concentration (g/kg)
dem_col = "htgy_DEM"       # Digital Elevation Model (elevation)
landuse_col = "LANDUSE"    # Land use
region_col = "Region"      # (Available but not used later)
slope_col = "SLOPE"        # Slope values
k1_col = "SOC_k1_fast_pool (1/month)"  # Fast pool decay rate
k2_col = "SOC_k2_slow_pool (1/month)"  # Slow pool decay rate

# -----------------------------------------------------------------------------
# Function: create_grid
# Creates a 2D grid using a pivot operation:
#   - Rows are unique latitudes (sorted in descending order)
#   - Columns are unique longitudes (sorted in ascending order)
# -----------------------------------------------------------------------------
def create_grid(data, col_name):
    return data.pivot(index=lat_col, columns=lon_col, values=col_name)\
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
C = np.clip(C, None, 13.8)  # Clip SOC values above 13.8
DEM = create_grid(df, dem_col)  # Elevation
SAND = create_grid(df, "SAND")
SILT = create_grid(df, "SILT")
CLAY = create_grid(df, "CLAY")
LANDUSE = create_grid(df, landuse_col)  # Land use grid (for later use)
REGION = create_grid(df, region_col)
SLOPE = create_grid(df, slope_col)
K_fast = create_grid(df, k1_col)  # Fast pool decay rate (1/month)
K_slow = create_grid(df, k2_col)  # Slow pool decay rate (1/month)
# Fill missing DEM and SLOPE values.
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

# =============================================================================
# 4) RUSLE COMPONENTS (MONTHLY)
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
    Sources: Gao et al. (2016); Liu et al. (2001).
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

K_factor = np.full_like(C, 0.03)
LS_factor = calculate_ls_factor(SLOPE)
P_factor = np.array([[calculate_p_factor(LANDUSE[i, j]) for j in range(LANDUSE.shape[1])]
                      for i in range(LANDUSE.shape[0])])

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
# NEW: RASTERIZE RIVER BASIN BOUNDARIES AND MAIN RIVER
# =============================================================================
# Here we add the effect of basin boundaries and rivers.
# - The small basin boundary (black) is read from "htgy_River_Basin.shp".
# - The larger basin boundary (green) is read from "94_area.shp".
# - The main river (red) is read from "ChinaRiver_main.shp".
# Because buffering in a geographic CRS is inaccurate, we first reproject the shapefiles
# to a projected CRS (EPSG:3857), perform buffering, then reproject back to our desired CRS (EPSG:4326).

dx = np.mean(np.diff(grid_x))
dy = np.mean(np.diff(grid_y))
resolution = np.mean([dx, dy])
buffer_distance = resolution  # Use 1× resolution for buffering

# Create a meshgrid of cell centers.
X, Y = np.meshgrid(grid_x, grid_y)

# Read the shapefiles.
small_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "htgy_River_Basin.shp")
large_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "94_area.shp")
river_shp = gpd.read_file(DATA_DIR / "China_River" / "ChinaRiver_main.shp")

# --- Filter river shapefile using the Loess Plateau border ---
# Reproject river shapefile to desired CRS and filter by intersection.
river_shp = river_shp.to_crs(desired_crs)
river_shp = river_shp[river_shp.intersects(loess_border_geom)]

# Reproject shapefiles to a projected CRS for buffering.
proj_crs = "EPSG:3857"
small_boundary_proj = small_boundary_shp.to_crs(proj_crs)
large_boundary_proj = large_boundary_shp.to_crs(proj_crs)
river_proj = river_shp.to_crs(proj_crs)

# Buffer the line features in the projected CRS.
small_boundary_buffered_proj = small_boundary_proj.buffer(buffer_distance)
large_boundary_buffered_proj = large_boundary_proj.buffer(buffer_distance)
river_buffered_proj = river_proj.buffer(buffer_distance)

# Union the buffered geometries and reproject back to the desired CRS.
small_boundary_union = gpd.GeoSeries(small_boundary_buffered_proj.union_all(), crs=proj_crs)\
                         .to_crs(desired_crs).iloc[0]
large_boundary_union = gpd.GeoSeries(large_boundary_buffered_proj.union_all(), crs=proj_crs)\
                         .to_crs(desired_crs).iloc[0]
river_union = gpd.GeoSeries(river_buffered_proj.union_all(), crs=proj_crs)\
              .to_crs(desired_crs).iloc[0]

# Create boolean mask arrays using np.vectorize.
small_boundary_mask = np.vectorize(lambda x, y: small_boundary_union.intersects(Point(x, y)))(X, Y)
large_boundary_mask = np.vectorize(lambda x, y: large_boundary_union.intersects(Point(x, y)))(X, Y)
# We will replace the river_mask test in the loop with prepared geometry.
river_mask = np.vectorize(lambda x, y: river_union.intersects(Point(x, y)))(X, Y)

# -----------------------------------------------------------------------------
# Helper: Compute outlet mask for a boundary.
# This function marks the cell with the lowest DEM among those flagged in the mask.
# -----------------------------------------------------------------------------
def compute_outlet_mask(boundary_mask, DEM):
    outlet_mask = np.zeros_like(boundary_mask, dtype=bool)
    indices = np.where(boundary_mask)
    if len(indices[0]) > 0:
        min_index = np.argmin(DEM[boundary_mask])
        outlet_i = indices[0][min_index]
        outlet_j = indices[1][min_index]
        outlet_mask[outlet_i, outlet_j] = True
    return outlet_mask

small_outlet_mask = compute_outlet_mask(small_boundary_mask, DEM)
large_outlet_mask = compute_outlet_mask(large_boundary_mask, DEM)

# --- Prepare the unioned river geometry for fast intersection testing ---
prepared_river = prep(river_union)
# (Optionally, you can also prepare the boundaries if needed)
prepared_small_boundary = prep(small_boundary_union)
prepared_large_boundary = prep(large_boundary_union)

# =============================================================================
# 6) ROUTE SOIL AND SOC FROM HIGHER CELLS (Modified with basin & river effects)
# =============================================================================
def distribute_soil_and_soc_with_dams(E_tcell, S, DEM, dam_positions, grid_x, grid_y,
                                      small_boundary_mask, small_outlet_mask,
                                      large_boundary_mask, large_outlet_mask,
                                      river_mask):
    """
    Route soil and its associated SOC from upstream cells while considering:
      - Basin boundaries: SOC will not cross small or large basin boundaries unless
        the current cell is the designated outlet.
      - Rivers: SOC flowing into a river cell is considered lost.

    Parameters:
      E_tcell : 2D array (t/cell/month) of local soil loss.
      S       : 2D array (kg/cell/month) of local SOC erosion.
      DEM     : 2D array of cell elevations.
      dam_positions : list of dicts with keys 'i', 'j', 'capacity_remained_tons'.
      grid_x, grid_y: 1D arrays of cell center coordinates.
      small_boundary_mask: Boolean array marking cells on the small basin boundary.
      small_outlet_mask: Boolean array marking the outlet cell for the small basin.
      large_boundary_mask: Boolean array marking cells on the larger basin boundary.
      large_outlet_mask: Boolean array marking the outlet cell for the larger basin.
      river_mask: Boolean array where True indicates a river cell.

    Returns:
      D_soil  : 2D array (t/cell/month) of soil deposited (upstream inflow).
      D_soc   : 2D array (kg/cell/month) of SOC deposited (upstream inflow).
      inflow_soil, inflow_soc: 2D arrays of soil and SOC still flowing downstream.
      lost_soc: 2D array (kg/cell/month) of SOC lost to rivers.
    """
    rows, cols = DEM.shape
    dam_capacity_map = {(dam['i'], dam['j']): dam['capacity_remained_tons'] for dam in dam_positions}
    inflow_soil = np.zeros_like(E_tcell, dtype=float)
    inflow_soc = np.zeros_like(E_tcell, dtype=float)
    D_soil = np.zeros_like(E_tcell, dtype=float)
    D_soc = np.zeros_like(E_tcell, dtype=float)
    lost_soc = np.zeros_like(E_tcell, dtype=float)  # Initialize lost SOC array

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
                    # Use the prepared river geometry for the neighbor.
                    if prepared_river.intersects(Point(grid_x[nj], grid_y[ni])):
                        lost_soc[i, j] += source_soc * ((DEM[i, j] - DEM[ni, nj]) / (np.hypot(di, dj) + 1e-9))
                        continue
                    # Check small basin boundary: if current and neighbor differ,
                    # allow crossing only if current cell is the small outlet.
                    if small_boundary_mask[i, j] != small_boundary_mask[ni, nj]:
                        if not small_outlet_mask[i, j]:
                            continue
                    # Check large basin boundary similarly.
                    if large_boundary_mask[i, j] != large_boundary_mask[ni, nj]:
                        if not large_outlet_mask[i, j]:
                            continue
                    # Allow flow only if neighbor DEM is lower.
                    if DEM[ni, nj] < DEM[i, j]:
                        slope_diff = (DEM[i, j] - DEM[ni, nj]) / (np.hypot(di, dj) + 1e-9)
                        neighbors.append(((ni, nj), slope_diff))
        total_slope = sum(s for _, s in neighbors)
        if total_slope > 0:
            for ((ni, nj), slope_val) in neighbors:
                fraction = slope_val / total_slope
                inflow_soil[ni, nj] += source_soil * fraction
                inflow_soc[ni, nj] += source_soc * fraction
    return D_soil, D_soc, inflow_soil, inflow_soc, lost_soc

# =============================================================================
# 7) VEGETATION INPUT & UPDATED SOC DYNAMIC MODEL
# =============================================================================
def vegetation_input(LAI):
    """
    Compute vegetation input based on LAI using an empirical formula.
    """
    return 0.0473 - 0.0913 * LAI + 0.0384 * LAI**2

def soc_dynamic_model(C_fast, C_slow,
                      soc_loss_g_kg_month, D_soil, D_soc, V,
                      K_fast, K_slow, p_fast_grid, dt, M_soil, lost_soc):
    """
    Update the SOC pools (g/kg) for one month.

    Deposited SOC (D_soc, in kg) is converted to a concentration increment (g/kg)
    using: deposition_concentration = (D_soc * 1000) / M_soil.

    Additionally, SOC lost to rivers (lost_soc) is subtracted from the pools.
    We assume the lost SOC is partitioned proportionally between the fast and slow pools.

    Parameters:
      (Same as before) plus:
      lost_soc : 2D array (kg/cell/month) representing SOC lost to rivers.
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

    # Partition lost SOC proportionally into fast and slow pools.
    lost_fast = lost_soc * p_fast_grid
    lost_slow = lost_soc * (1 - p_fast_grid)

    # Update SOC pools with all fluxes.
    C_fast_new = np.maximum(
        C_fast + (erosion_fast + deposition_fast + vegetation_fast + reaction_fast - lost_fast) * dt, 0)
    C_slow_new = np.maximum(
        C_slow + (erosion_slow + deposition_slow + vegetation_slow + reaction_slow - lost_slow) * dt, 0)

    return C_fast_new, C_slow_new

# =============================================================================
# 8) HELPER: REGRID CMIP6 POINT DATA TO 2D GRID
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
M_soil = 1.0e8  # Total soil mass per cell (kg)

# (The remainder of your code uses M_soil to convert deposited SOC mass (kg/cell/month)
# into a concentration increment (g/kg/month) via the formula:
#    deposition_concentration (g/kg) = (D_soc * 1000) / M_soil
# )

# Initialize current SOC pools with the initial values.
C_fast_current = C_fast.copy()
C_slow_current = C_slow.copy()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / "Figure", exist_ok=True)
os.makedirs(OUTPUT_DIR / "Data", exist_ok=True)

for year in range(start_year, end_year + 1):
    # --- NEW: Filter only dams built on or before the current year.
    df_dam_active = df_dam[df_dam["year"] <= year].copy()

    # Build a fresh list of dam positions.
    dam_positions_active = []
    for _, row in df_dam_active.iterrows():
        i_idx = find_nearest_index(grid_y, row["y"])
        j_idx = find_nearest_index(grid_x, row["x"])
        capacity_10000_m3 = row["capacity_remained"]  # in units of 10,000 m³.
        capacity_tons = capacity_10000_m3 * 10000 * BULK_DENSITY
        dam_positions_active.append({
            'i': i_idx,
            'j': j_idx,
            'capacity_remained_tons': capacity_tons
        })

    # Open the NetCDF file for the current year.
    nc_file = PROCESSED_DIR / "ERA5_Data_Monthly_Resampled" / f"resampled_{year}.nc"
    if not os.path.exists(nc_file):
        print(f"NetCDF file not found for year {year}: {nc_file}")
        continue

    with nc.Dataset(nc_file) as ds:
        valid_time = ds.variables['valid_time'][:]  # Expecting 12 months.
        n_time = len(valid_time)
        lon_nc = ds.variables['longitude'][:]
        lat_nc = ds.variables['latitude'][:]
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
            # Route soil and SOC from higher (upstream) cells with basin and river effects.
            # -------------------------------------------------------------------------
            D_soil, D_soc, inflow_soil, inflow_soc, lost_soc = distribute_soil_and_soc_with_dams(
                E_tcell_month, S, DEM, dam_positions_active, grid_x, grid_y,
                small_boundary_mask, small_outlet_mask,
                large_boundary_mask, large_outlet_mask,
                river_mask
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

            deposition_concentration = (D_soc * 1000.0) / M_soil
            deposition_fast = deposition_concentration * p_fast_grid
            deposition_slow = deposition_concentration * (1 - p_fast_grid)

            vegetation_fast = V * p_fast_grid
            vegetation_slow = V * (1 - p_fast_grid)

            reaction_fast = -K_fast * C_fast_current
            reaction_slow = -K_slow * C_slow_current

            # -------------------------------------------------------------------------
            # Update SOC pools using the dynamic model.
            # Now subtract the lost SOC (partitioned by p_fast_grid).
            # -------------------------------------------------------------------------
            C_fast_current, C_slow_current = soc_dynamic_model(
                C_fast_current, C_slow_current,
                SOC_loss_g_kg_month, D_soil, D_soc, V,
                K_fast, K_slow, p_fast_grid,
                dt=1,  # 1-month timestep.
                M_soil=M_soil,
                lost_soc=lost_soc  # Subtract river loss.
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
            filename_fig = f"SOC_{year}_{month_idx + 1:02d}_timestep_{global_timestep}_River.png"
            plt.savefig(os.path.join(OUTPUT_DIR / "Figure", filename_fig))
            plt.close(fig)

            # -------------------------------------------------------------------------
            # (B) SAVE DATA OUTPUT AS CSV.
            # Save each term in the SOC balance equation along with E_t_ha_month and landuse.
            # Now include a column for SOC lost to rivers.
            # -------------------------------------------------------------------------
            rows_grid, cols_grid = C_fast_current.shape
            lat_list = []
            lon_list = []
            landuse_list = []  # Landuse type
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
            lost_soc_list = []  # New list for SOC lost to river

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
                    lost_soc_list.append(lost_soc[i, j])  # Save lost SOC for this cell

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
                'E_t_ha_month': E_t_ha_list,
                'Lost_SOC_River': lost_soc_list  # New column for SOC lost to rivers
            })

            filename_csv = f"SOC_terms_{year}_{month_idx + 1:02d}_timestep_{global_timestep}_River.csv"
            df_out.to_csv(os.path.join(OUTPUT_DIR / "Data", filename_csv), index=False)
            print(f"Saved CSV output for Year {year}, Month {month_idx + 1} as {filename_csv}")

print("Simulation complete. Final SOC distribution is in C_fast_current + C_slow_current.")
