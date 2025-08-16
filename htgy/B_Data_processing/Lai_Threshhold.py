#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute yearly LAI thresholds (via Otsu) and corresponding SOC statistics
(spatial mean, spatial std of annual means, & mean of within-year stds)
for all years 1950–2024.
For 1950–2006, SOC is read from a single NetCDF; for 2007–2024, SOC is read
from monthly Parquet files. Results are saved to a CSV.
"""
import os
import sys
from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu  # pip install scikit-image

# ─── adjust path so globals.py can be imported ───────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR

# ─── configuration ───────────────────────────────────────────────────────────
START_YEAR  = 1950
END_YEAR    = 2024
LAI_FILES = {
    "past":    Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_1950-2000.nc",
    "present": Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2001-2014.nc",
    "future":  Path(PROCESSED_DIR) / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_245.nc",
}
PAST_SOC_NC   = Path(OUTPUT_DIR) / "Data" / "SOC_Past 2" / "Total_C_1950-2007_monthly.nc"
PRESENT_SOC_DIR = Path(OUTPUT_DIR) / "Data" / "SOC_Present 6"

# preload the past SOC netCDF
ds_past = xr.open_dataset(PAST_SOC_NC)

results = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"Processing year {year}…")
    # ─── 1) select LAI file & compute annual-mean LAI ────────────────────────
    if year <= 2000:
        lai_path = LAI_FILES["past"]
    elif year <= 2014:
        lai_path = LAI_FILES["present"]
    else:
        lai_path = LAI_FILES["future"]

    ds_lai   = xr.open_dataset(lai_path)
    lai_sel  = ds_lai["lai"].sel(time=slice(f"{year}-01", f"{year}-12"))
    lai_mean = lai_sel.mean(dim="time")
    lats     = ds_lai["lat"].values
    lons     = ds_lai["lon"].values
    lai_vals = lai_mean.values
    ds_lai.close()

    # Otsu threshold
    valid    = ~np.isnan(lai_vals)
    thr      = threshold_otsu(lai_vals[valid])

    # build DataFrame of LAI and group assignment
    # lats, lons and lai_vals are all 1-D arrays of length N_points
    lai_df = pd.DataFrame({
        "LAT": lats,
        "LON": lons,
        "LAI": lai_vals
    }).dropna(subset=["LAI"])

    lai_df["group"] = np.where(lai_df["LAI"] > thr, "High_LAI", "Low_LAI")

    # ─── 2) read SOC for this year ───────────────────────────────────────────
    if year <= 2006:
        # from past netCDF
        ds_sel = ds_past.sel(time=slice(f"{year}-01-01", f"{year}-12-01"))
        soc_mean_da = ds_sel["total_C"].mean(dim="time")
        soc_std_da  = ds_sel["total_C"].std(dim="time")
        # assume dims are (time, lat, lon)
        lat_vals = soc_mean_da["lat"].values
        lon_vals = soc_mean_da["lon"].values
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        soc_stats = pd.DataFrame({
            "LAT": lat2d.flatten(),
            "LON": lon2d.flatten(),
            "annual_mean_SOC": soc_mean_da.values.flatten(),
            "annual_std_SOC": soc_std_da.values.flatten()
        }).dropna(subset=["annual_mean_SOC"])
    else:
        # from monthly Parquet (present)
        frames = []
        for mo in range(1, 13):
            fp = PRESENT_SOC_DIR / f"SOC_terms_{year}_{mo:02d}_River.parquet"
            df = pd.read_parquet(fp)
            df = df[df["Total_C"].notna()][["LAT","LON","Total_C"]]
            frames.append(df)
        soc_all = pd.concat(frames, ignore_index=True)
        soc_stats = (
            soc_all
            .groupby(["LAT","LON"])["Total_C"]
            .agg(annual_mean_SOC="mean", annual_std_SOC="std")
            .reset_index()
        )

    # ─── 3) merge LAI & SOC on rounded coords ────────────────────────────────
    for df_ in (lai_df, soc_stats):
        df_["lat_r"] = df_["LAT"].round(4)
        df_["lon_r"] = df_["LON"].round(4)

    merged = pd.merge(
        lai_df, soc_stats,
        on=["lat_r","lon_r"],
        how="inner",
        suffixes=("_lai","_soc")
    )

    # ─── 4) compute group stats ───────────────────────────────────────────────
    # spatial summary of per-point annual means
    spatial = merged.groupby("group")["annual_mean_SOC"] \
                    .agg(mean_SOC="mean", spatial_std_SOC="std")
    # temporal summary of per-point within-year stds
    temporal = merged.groupby("group")["annual_std_SOC"] \
                     .agg(temporal_std_SOC="mean")

    for grp in ["High_LAI","Low_LAI"]:
        results.append({
            "year":            year,
            "lai_threshold":   thr,
            "group":           grp,
            "mean_SOC":        spatial.loc[grp,"mean_SOC"],
            "spatial_std_SOC": spatial.loc[grp,"spatial_std_SOC"],
            "temporal_std_SOC": temporal.loc[grp,"temporal_std_SOC"],
        })

# close the past dataset
ds_past.close()

# ─── 5) save all years’ results to CSV ─────────────────────────────────────
out_df = pd.DataFrame(results)
out_fp = Path(OUTPUT_DIR) / "lai_soc_summary_1950_2024.csv"
out_df.to_csv(out_fp, index=False)
print(f"\nAll done — results saved to:\n  {out_fp}")
