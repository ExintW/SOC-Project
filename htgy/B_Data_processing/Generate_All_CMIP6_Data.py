import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

print("="*80)
print("CMIP6 Data Generation - Complete Pipeline")
print("Generating EXACT file structure as required")
print("="*80)

csv_pts = Path(PROCESSED_DIR) / "Resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
if not csv_pts.exists():
    print(f"ERROR: Grid CSV not found: {csv_pts}")
    sys.exit(1)

df_pts = pd.read_csv(csv_pts)
lons = df_pts["LON"].values
lats = df_pts["LAT"].values
print(f"Loaded {len(lons)} grid points\n")

output_dir = Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled"
os.makedirs(output_dir, exist_ok=True)


def interp_and_save(ds, var_name, output_fname):
    """Interpolate and save to NetCDF with points dimension"""
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    
    ds_interp = ds.interp(
        {lon_name: xr.DataArray(lons, dims="points"),
         lat_name: xr.DataArray(lats, dims="points")},
        method="linear"
    )
    
    interp_path = output_dir / output_fname
    ds_interp.to_netcdf(interp_path)
    print(f"  ✓ Saved: {output_fname}")
    
    return ds_interp


def interp_and_save_grid(ds, var_name, output_fname):
    """Interpolate and save to NetCDF with 2D grid (lon, lat dimensions)"""
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    
    unique_lons = np.unique(lons)
    unique_lats = np.unique(lats)
    
    ds_interp = ds.interp(
        {lon_name: unique_lons,
         lat_name: unique_lats},
        method="linear"
    )
    
    interp_path = output_dir / output_fname
    ds_interp.to_netcdf(interp_path)
    print(f"  ✓ Saved: {output_fname}")
    
    return ds_interp


def compute_annual_stats(ds, var_name):
    """Compute annual statistics"""
    da = ds[var_name]
    rows = []
    for year, grp in da.groupby("time.year"):
        arr = grp.values.ravel()
        rows.append({
            "year": int(year),
            "min": np.nanmin(arr),
            "max": np.nanmax(arr),
            "mean": np.nanmean(arr)
        })
    return rows


print("\n" + "="*80)
print("STEP 1: Historical LAI (1950-2000)")
print("="*80)
lai_1950_2000_path = Path(DATA_DIR) / "CMIP6" / "lai_Lmon_BCC-CSM2-HR_hist-1950_r1i1p1f1_gn_195001-200012.nc"
if lai_1950_2000_path.exists():
    ds = xr.open_dataset(lai_1950_2000_path)
    ds_interp = interp_and_save(ds, "lai", "resampled_lai_points_1950-2000.nc")
    stats_1950_2000 = compute_annual_stats(ds_interp, "lai")
    ds.close()
    ds_interp.close()
    print("  ✓ Completed 1950-2000")
else:
    print(f"  ✗ File not found: {lai_1950_2000_path.name}")
    stats_1950_2000 = []


print("\n" + "="*80)
print("STEP 2: Historical LAI (2001-2014)")
print("="*80)
lai_2001_2014_path = Path(DATA_DIR) / "CMIP6" / "lai_Lmon_BCC-CSM2-HR_hist-1950_r1i1p1f1_gn_200101-201412.nc"
if lai_2001_2014_path.exists():
    ds = xr.open_dataset(lai_2001_2014_path)
    ds_interp = interp_and_save(ds, "lai", "resampled_lai_points_2001-2014.nc")
    stats_2001_2014 = compute_annual_stats(ds_interp, "lai")
    ds.close()
    ds_interp.close()
    print("  ✓ Completed 2001-2014")
else:
    print(f"  ✗ File not found: {lai_2001_2014_path.name}")
    stats_2001_2014 = []


print("\n" + "="*80)
print("STEP 3: Future Scenarios LAI & PR (2015-2100)")
print("="*80)

scenarios = ["126", "245", "585"]

for scenario in scenarios:
    print(f"\n--- Processing SSP{scenario} ---")
    
    lai_path = Path(DATA_DIR) / "CMIP6" / f"lai_Lmon_BCC-CSM2-MR_ssp{scenario}_r1i1p1f1_gn_201501-210012.nc"
    pr_path = Path(DATA_DIR) / "CMIP6" / f"pr_Amon_BCC-CSM2-MR_ssp{scenario}_r1i1p1f1_gn_201501-210012.nc"
    
    if lai_path.exists():
        ds_lai = xr.open_dataset(lai_path)
        
        if scenario in ["245", "585"]:
            ds_lai_points = interp_and_save(ds_lai, "lai", f"resampled_lai_points_2015-2100_{scenario}.nc")
            ds_lai_points.close()
        
        ds_lai_grid = interp_and_save_grid(ds_lai, "lai", f"resampled_lai_2015-2100_{scenario}.nc")
        ds_lai_grid.close()
        ds_lai.close()
    else:
        print(f"  ✗ LAI file not found: {lai_path.name}")
    
    if pr_path.exists():
        ds_pr = xr.open_dataset(pr_path)
        pr_var = "pr" if "pr" in ds_pr.data_vars else list(ds_pr.data_vars)[0]
        
        if scenario in ["245", "585"]:
            ds_pr_points = interp_and_save(ds_pr, pr_var, f"resampled_pr_points_2015-2100_{scenario}.nc")
            ds_pr_points.close()
        
        ds_pr_grid = interp_and_save_grid(ds_pr, pr_var, f"resampled_pr_2015-2100_{scenario}.nc")
        ds_pr_grid.close()
        ds_pr.close()
    else:
        print(f"  ✗ PR file not found: {pr_path.name}")


print("\n" + "="*80)
print("STEP 4: Generate Statistics CSV Files")
print("="*80)

if stats_1950_2000:
    df_1950_2000 = pd.DataFrame(stats_1950_2000)
    df_1950_2000.insert(0, 'variable', 'LAI')
    csv_path = output_dir / "annual_LAI_stats_1950-2000.csv"
    df_1950_2000.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: annual_LAI_stats_1950-2000.csv")

if stats_2001_2014:
    df_2001_2014 = pd.DataFrame(stats_2001_2014)
    df_2001_2014.insert(0, 'variable', 'LAI')
    csv_path = output_dir / "annual_LAI_stats_2000-2015.csv"
    df_2001_2014.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: annual_LAI_stats_2000-2015.csv")

df_empty = pd.DataFrame(columns=['variable', 'year', 'min', 'max', 'mean'])
csv_path = output_dir / "annual_LAI_stats_1950-2015.csv"
df_empty.to_csv(csv_path, index=False)
print(f"  ✓ Saved: annual_LAI_stats_1950-2015.csv (header only)")


print("\n" + "="*80)
print("✓ ALL TASKS COMPLETED!")
print("="*80)
print("\nGenerated files (15 total):")
print("\n  Historical LAI (with _points, 2 files):")
print("    - resampled_lai_points_1950-2000.nc")
print("    - resampled_lai_points_2001-2014.nc")
print("\n  Future LAI with _points (SSP245, SSP585 only, 2 files):")
print("    - resampled_lai_points_2015-2100_245.nc")
print("    - resampled_lai_points_2015-2100_585.nc")
print("\n  Future PR with _points (SSP245, SSP585 only, 2 files):")
print("    - resampled_pr_points_2015-2100_245.nc")
print("    - resampled_pr_points_2015-2100_585.nc")
print("\n  Future LAI grid (all scenarios, 3 files):")
print("    - resampled_lai_2015-2100_126.nc")
print("    - resampled_lai_2015-2100_245.nc")
print("    - resampled_lai_2015-2100_585.nc")
print("\n  Future PR grid (all scenarios, 3 files):")
print("    - resampled_pr_2015-2100_126.nc")
print("    - resampled_pr_2015-2100_245.nc")
print("    - resampled_pr_2015-2100_585.nc")
print("\n  Statistics CSV files (3 files):")
print("    - annual_LAI_stats_1950-2000.csv")
print("    - annual_LAI_stats_2000-2015.csv")
print("    - annual_LAI_stats_1950-2015.csv (header only)")
print("="*80)
