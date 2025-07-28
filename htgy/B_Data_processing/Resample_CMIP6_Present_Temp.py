import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# Append project root so we can import DATA_DIR, PROCESSED_DIR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# Define scenarios and corresponding input filenames
SCENARIOS = {
    "hist": "tsl_Lmon_BCC-CSM2-MR_historical_r1i1p1f1_gn_193001-201412.nc",
    "126":  "tsl_Lmon_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_201501-210012.nc",
    "245":  "tsl_Lmon_BCC-CSM2-MR_ssp245_r1i1p1f1_gn_201501-210012.nc",
    "370":  "tsl_Lmon_BCC-CSM2-MR_ssp370_r1i1p1f1_gn_201501-210012.nc",
    "585":  "tsl_Lmon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_201501-210012.nc"
}

# Load 1km grid point coordinates
csv_pts = Path(PROCESSED_DIR) / "resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
df_pts = pd.read_csv(csv_pts)
lons = df_pts["LON"].values
lats = df_pts["LAT"].values

# Output directory
output_dir = Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled"
os.makedirs(output_dir, exist_ok=True)

def interp_and_collect_tsl(ds, var_name, label, save_interp_fname):
    """
    Interpolate ds[var_name] (top 5 layers averaged) onto target points bilinearly,
    save the interpolated dataset, and return annual stats rows.
    """
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    lat_name = "latitude"  if "latitude"  in ds.dims else "lat"
    depth_dim = "depth"

    # Average over top 5 depth levels (~0–0.6 m)
    ds = ds.sel({depth_dim: ds[depth_dim][:5]})
    ds[var_name] = ds[var_name].mean(dim=depth_dim)

    # Interpolation
    ds_interp = ds.interp(
        {lon_name: xr.DataArray(lons, dims="points"),
         lat_name: xr.DataArray(lats, dims="points")},
        method="linear"
    )

    # Save interpolated NetCDF
    interp_path = output_dir / save_interp_fname
    ds_interp.to_netcdf(interp_path)
    print(f"→ Saved interpolated {label} to {interp_path.name}")

    # Annual stats
    da = ds_interp[var_name]
    rows = []
    for year, grp in da.groupby("time.year"):
        arr = grp.values.ravel()
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        mu = np.nanmean(arr)
        print(f"{label} {int(year)} → min={mn:.3f}, max={mx:.3f}, mean={mu:.3f}")
        rows.append((label, int(year), mn, mx, mu))

    ds_interp.close()
    return rows

# Loop over each scenario
for scn, fname in SCENARIOS.items():
    tsl_path = Path(DATA_DIR) / "CMIP6" / fname
    try:
        print(f"\n>>> Processing Scenario: {'Historical' if scn == 'hist' else 'SSP' + scn}")
        ds_tsl = xr.open_dataset(tsl_path)

        label = f"TSL_top5layers_Historical" if scn == "hist" else f"TSL_top5layers_SSP{scn}"
        interp_file = f"resampled_tsl_top5layers_points_{scn}_interp.nc"
        csv_file    = output_dir / f"annual_TSL_top5layers_stats_{scn}.csv"

        # Interpolate + compute stats
        tsl_rows = interp_and_collect_tsl(
            ds_tsl,
            var_name="tsl",
            label=label,
            save_interp_fname=interp_file
        )
        ds_tsl.close()

        # Save CSV
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["variable", "year", "min (°C)", "max (°C)", "mean (°C)"])
            for row in tsl_rows:
                writer.writerow(row)
        print(f"→ Annual stats saved to {csv_file.name}")

        # Visualization
        interp_nc = output_dir / interp_file
        ds_res = xr.open_dataset(interp_nc)
        mean_tsl = ds_res["tsl"].mean(dim="time")

        lon_var = "longitude" if "longitude" in ds_res.coords else "lon"
        lat_var = "latitude" if "latitude" in ds_res.coords else "lat"

        lons_plot = ds_res[lon_var].values
        lats_plot = ds_res[lat_var].values

        plt.figure(figsize=(10, 6))
        sc = plt.scatter(
            lons_plot, lats_plot,
            c=mean_tsl,
            s=10,
            edgecolor="none"
        )
        title_label = "Historical" if scn == "hist" else f"SSP{scn}"
        plt.colorbar(sc, label=f"Mean TSL (°C, top 5 layers, {title_label})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Mean TSL (Top 5 layers, {title_label})")
        plt.tight_layout()
        plt.savefig(output_dir / f"mean_TSL_top5layers_map_{scn}.png", dpi=300)
        plt.close()

        ds_res.close()

    except Exception as e:
        print(f"⚠️ Error in {scn}: {e}")
