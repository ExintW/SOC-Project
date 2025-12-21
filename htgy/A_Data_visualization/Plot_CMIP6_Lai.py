import os
import sys
from pathlib import Path

import xarray as xr
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---- adjust this to your project structure ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *
# ---------------------------------------------

# Define output directory for the maps
out_dir = OUTPUT_DIR / "Lai_Graphs"
out_dir.mkdir(parents=True, exist_ok=True)

# Path to Loess Plateau border shapefile
loess_border_path = Path(DATA_DIR) / "Loess_Plateau_vector_border.shp"

# File mappings
files = {
    "historical": "resampled_lai_points_1950-2000.nc",
    "present":    "resampled_lai_points_2001-2014.nc",
    "ssp126":     "resampled_lai_points_2015-2100_126.nc",
    "ssp245":     "resampled_lai_points_2015-2100_245.nc",
    "ssp370":     "resampled_lai_points_2015-2100_370.nc",
    "ssp585":     "resampled_lai_points_2015-2100_585.nc",
}

# Decadal years
hist_years   = list(range(1950, 2011, 10))  # 1950, 60, ..., 2010
future_years = list(range(2020, 2101, 10))  # 2020, 30, ..., 2100
scenarios    = ["ssp126", "ssp245", "ssp370", "ssp585"]

# Load Loess Plateau border
border = gpd.read_file(loess_border_path)
minx, miny, maxx, maxy = border.total_bounds


def plot_lai_map(ds, year, title, out_path):
    # average over the 12 months of that year
    lai_yr = ds["lai"].sel(time=slice(f"{year}-01-01", f"{year}-12-31")).mean(dim="time")

    # convert to GeoDataFrame
    lon = lai_yr["lon"].values
    lat = lai_yr["lat"].values
    vals = lai_yr.values
    df = pd.DataFrame({"lon": lon, "lat": lat, "lai": vals})
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=border.crs)

    # clip to Loess Plateau
    clipped = gpd.clip(gdf, border)

    # plot
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05

    fig, ax = plt.subplots(figsize=(16, 8))
    # border.boundary.plot(ax=ax, linewidth=0.5, color="black")

    clipped.plot(
        column="lai",
        ax=ax,
        markersize=5,
        legend=False
    )

    # colorbar axis aligned with map axis height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.2)
    mappable = ax.collections[-1]
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label("LAI")

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _get_inside_point_indices(ds, border_gdf):
    lon = ds["lon"].values
    lat = ds["lat"].values
    pts_df = pd.DataFrame({"lon": lon, "lat": lat})
    pts_gdf = gpd.GeoDataFrame(
        pts_df,
        geometry=gpd.points_from_xy(pts_df.lon, pts_df.lat),
        crs=border_gdf.crs
    )
    clipped_pts = gpd.clip(pts_gdf, border_gdf)
    if clipped_pts.empty:
        raise ValueError("No LAI points fall inside the Loess Plateau border. Check CRS or coordinates.")
    return clipped_pts.index.to_numpy()


def compute_loess_mean_annual_lai(ds, border_gdf):
    """
    Return a pandas.Series: index = year (int), value = annual mean LAI
    Works with cftime calendars by doing groupby in xarray.
    Assumes ds['lai'] dims: (time, points) and ds['lon'], ds['lat'] align with points.
    """
    inside_idx = _get_inside_point_indices(ds, border_gdf)

    lai_monthly_mean = ds["lai"].isel(points=inside_idx).mean(dim="points")
    lai_annual = lai_monthly_mean.groupby("time.year").mean(dim="time")

    years = lai_annual["year"].values
    values = lai_annual.values
    return pd.Series(values, index=years, name="lai")


def compute_high_low_annual_series(hist_ds, pres_ds, border_gdf, threshold=1.231):
    """
    Make two annual mean LAI time series (1950 to 2014):
      - High zone: points whose long term mean LAI (1950 to 2014) > threshold
      - Low zone:  points whose long term mean LAI (1950 to 2014) <= threshold

    Classification is fixed using the long term average, which matches "based on the average".
    Returns: (annual_high_series, annual_low_series)
    """
    # Use indices from one dataset (same point layout)
    inside_idx = _get_inside_point_indices(hist_ds, border_gdf)

    # Combine monthly LAI (cftime safe), then restrict to inside Loess Plateau
    lai_all = xr.concat([hist_ds["lai"], pres_ds["lai"]], dim="time")
    lai_all = lai_all.isel(points=inside_idx)

    # Long term mean per point (1950 to 2014) then build fixed masks
    clim_mean = lai_all.mean(dim="time")  # dims: points
    high_mask = clim_mean > threshold
    low_mask = ~high_mask

    # Monthly spatial mean for each fixed zone
    lai_high_monthly = lai_all.where(high_mask).mean(dim="points", skipna=True)
    lai_low_monthly  = lai_all.where(low_mask).mean(dim="points", skipna=True)

    # Annual mean (cftime safe)
    lai_high_annual = lai_high_monthly.groupby("time.year").mean(dim="time")
    lai_low_annual  = lai_low_monthly.groupby("time.year").mean(dim="time")

    high_series = pd.Series(lai_high_annual.values, index=lai_high_annual["year"].values, name="lai_high")
    low_series  = pd.Series(lai_low_annual.values,  index=lai_low_annual["year"].values,  name="lai_low")

    # Keep exact range
    high_series = high_series.loc[1950:2014]
    low_series  = low_series.loc[1950:2014]

    return high_series, low_series


def plot_annual_lai_series(annual_series, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(annual_series.index, annual_series.values)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean LAI")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_high_low_series(high_series, low_series, threshold, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(high_series.index, high_series.values, label=f"Zone mean where long term mean > {threshold}")
    ax.plot(low_series.index,  low_series.values,  label=f"Zone mean where long term mean <= {threshold}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean LAI")
    ax.set_title(f"Loess Plateau Mean Annual LAI by Fixed Zones (threshold {threshold})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Plot average LAI from 1950 to 2014 (Loess Plateau mean)
# ------------------------------------------------------------
hist_ds = xr.open_dataset(PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / files["historical"])
pres_ds = xr.open_dataset(PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / files["present"])

annual_hist = compute_loess_mean_annual_lai(hist_ds, border)   # 1950-2000
annual_pres = compute_loess_mean_annual_lai(pres_ds, border)   # 2001-2014

annual_1950_2014 = pd.concat([annual_hist, annual_pres]).sort_index()
annual_1950_2014 = annual_1950_2014.loc[1950:2014]

series_out = out_dir / "lai_loessplateau_mean_annual_1950_2014.png"
plot_annual_lai_series(
    annual_1950_2014,
    "Loess Plateau Mean Annual LAI (1950-2014)",
    series_out
)

# ------------------------------------------------------------
# NEW: plot two curves split by fixed zones using threshold 1.231
# ------------------------------------------------------------
THRESH = 1.231
high_series, low_series = compute_high_low_annual_series(hist_ds, pres_ds, border, threshold=THRESH)

split_out = out_dir / f"lai_loessplateau_mean_annual_split_{THRESH}.png"
plot_high_low_series(high_series, low_series, THRESH, split_out)

# Optional: save the split series to CSV
split_csv = out_dir / f"lai_loessplateau_mean_annual_split_{THRESH}.csv"
pd.DataFrame({
    "year": high_series.index.astype(int),
    "lai_high": high_series.values,
    "lai_low": low_series.values
}).to_csv(split_csv, index=False)

hist_ds.close()
pres_ds.close()

# Generate maps for 1950-2010
for year in hist_years:
    src = "historical" if year <= 2000 else "present"
    ds = xr.open_dataset(PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / files[src])
    out_fp = out_dir / f"lai_{year}.png"
    plot_lai_map(ds, year, f"LAI {year}", out_fp)
    ds.close()

# Generate decadal maps for each future scenario
for scen in scenarios:
    ds = xr.open_dataset(PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / files[scen])
    for year in future_years:
        out_fp = out_dir / f"lai_{scen}_{year}.png"
        plot_lai_map(ds, year, f"{scen.upper()} LAI {year}", out_fp)
    ds.close()

print(f"Saved all LAI maps to: {out_dir}")
print(f"Saved LAI annual mean series to: {series_out}")
print(f"Saved split high-low series plot to: {split_out}")
print(f"Saved split high-low series CSV to: {split_csv}")
