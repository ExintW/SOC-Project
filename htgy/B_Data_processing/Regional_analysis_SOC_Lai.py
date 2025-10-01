#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Yearly per-region (A,B,C,D) SOC & LAI summary for 1950–2100.

Outputs one CSV:
  OUTPUT_DIR / "soc_lai_yearly_by_region_1950_2100.csv"

Columns: year, period(Past/Present/Future), scenario(Past/Present/sspXXX),
         REGION, soc_mean, soc_std, lai_mean
"""

import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

# ── project globals ───────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

# ── paths & config ───────────────────────────────────────────────────────────
# Regions (use the cleaned layer you just created)
REGIONS_PATH = DATA_DIR / "Ecological regionalization" / "Processed" / "Ecological_regionalization_REGIONS_clean.gpkg"
# Or use SHP:
# REGIONS_PATH = DATA_DIR / "Ecological regionalization" / "Processed" / "Ecological_regionalization_REGIONS_clean.shp"
REGION_FIELD  = "REGION"

# SOC sources
PAST_SOC_NC      = OUTPUT_DIR / "Data" / "SOC_Past 2" / "Total_C_1950-2007_monthly.nc"
PRESENT_SOC_DIR  = OUTPUT_DIR / "Data" / "SOC_Present 7"
FUTURE_SOC_DIR   = OUTPUT_DIR / "Data" / "SOC_Future 7"  # contains subdirs: 126,245,370,585

# LAI sources
LAI_PAST_FP      = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_1950-2000.nc"
LAI_PRESENT_FP   = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2001-2014.nc"
LAI_2015_245_FP  = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_245.nc"

# Optional: scenario-specific future LAI files (use if present)
FUTURE_SSP_LAI = {
    "126": PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_126.nc",
    "245": PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_245.nc",
    "370": PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_370.nc",
    "585": PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_585.nc",
}

OUT_CSV = OUTPUT_DIR / "soc_lai_yearly_by_region_1950_2100.csv"

YEARS_PAST    = range(1950, 2007)   # 1950–2006
YEARS_PRESENT = range(2007, 2025)   # 2007–2024
YEARS_FUTURE  = range(2025, 2101)   # 2025–2100
SSPS          = ["126", "245", "370", "585"]

# ── helpers: regions & spatial mapping ───────────────────────────────────────
def load_regions(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Regions file not found: {path}")
    gdf = gpd.read_file(path)
    if REGION_FIELD not in gdf.columns:
        raise KeyError(f"'{REGION_FIELD}' missing in regions")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(4326, allow_override=True) if gdf.crs is None else gdf.to_crs(4326)
    # Assume geometries already valid from your cleaning script
    return gdf[[REGION_FIELD, "geometry"]].copy()

def build_point_region_map_from_xy(lat_vals, lon_vals, regions: gpd.GeoDataFrame, tag: str) -> pd.DataFrame:
    """
    Build mapping LAT,LON -> REGION for a grid/point set (from arrays).
    Accepts either 1D or 2D lat/lon (will flatten).
    """
    if lat_vals.ndim == 1 and lon_vals.ndim == 1 and lat_vals.size != lon_vals.size:
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        pts = pd.DataFrame({"LAT": lat2d.ravel(), "LON": lon2d.ravel()})
    else:
        pts = pd.DataFrame({"LAT": np.ravel(lat_vals), "LON": np.ravel(lon_vals)})
    gpts = gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts["LON"].values, pts["LAT"].values),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gpts, regions, how="left", predicate="intersects")
    mapping = (
        joined[["LAT", "LON", REGION_FIELD]]
        .groupby(["LAT","LON"], sort=False).first()
        .reset_index()
    )
    matched = mapping[REGION_FIELD].notna().sum()
    print(f"[{tag} mapping] matched {matched}/{len(mapping)} points to regions")
    return mapping

def build_point_region_map_from_parquet(sample_fp: Path, regions: gpd.GeoDataFrame, tag: str) -> pd.DataFrame:
    df = pd.read_parquet(sample_fp, columns=["LAT", "LON"])
    pts = df.drop_duplicates().reset_index(drop=True)
    gpts = gpd.GeoDataFrame(
        pts, geometry=gpd.points_from_xy(pts["LON"].values, pts["LAT"].values), crs="EPSG:4326"
    )
    joined = gpd.sjoin(gpts, regions, how="left", predicate="intersects")
    mapping = (
        joined[["LAT", "LON", REGION_FIELD]]
        .groupby(["LAT","LON"], sort=False).first()
        .reset_index()
    )
    matched = mapping[REGION_FIELD].notna().sum()
    print(f"[{tag} mapping] matched {matched}/{len(mapping)} points to regions")
    return mapping

def find_one_parquet(dir_path: Path, years) -> Path:
    for y in years:
        for m in range(1, 13):
            fp = dir_path / f"SOC_terms_{y}_{m:02d}_River.parquet"
            if fp.exists():
                return fp
    raise FileNotFoundError(f"No parquet found under {dir_path}")

# ── LAI helpers ──────────────────────────────────────────────────────────────
def open_lai_dataset(path: Path | None, fallback: Path | None = None) -> xr.Dataset:
    """
    Open a LAI dataset; if path is None/missing, fall back to another path.
    """
    if path is not None and Path(path).exists():
        return xr.open_dataset(path)
    if fallback is not None and Path(fallback).exists():
        warnings.warn(f"LAI file {path} missing; falling back to {fallback.name}")
        return xr.open_dataset(fallback)
    raise FileNotFoundError(f"LAI dataset not found (path={path}, fallback={fallback})")

def extract_lai_latlon(ds: xr.Dataset):
    lat = ds.get("lat", None)
    lon = ds.get("lon", None)
    if lat is None or lon is None:
        # Try coords on the variable
        lat = ds["lai"].coords.get("lat", None)
        lon = ds["lai"].coords.get("lon", None)
    if lat is None or lon is None:
        raise KeyError("LAI dataset missing 'lat'/'lon'")
    return np.asarray(lat.values), np.asarray(lon.values)

def lai_annual_df(ds: xr.Dataset, year: int) -> pd.DataFrame:
    lai = ds["lai"].sel(time=slice(f"{year}-01", f"{year}-12")).mean(dim="time", skipna=True)
    lat_da = lai.coords.get("lat", ds.get("lat", None))
    lon_da = lai.coords.get("lon", ds.get("lon", None))
    if lat_da is None or lon_da is None:
        return pd.DataFrame({"LAT": [], "LON": [], "LAI": []})
    lat_vals = np.asarray(lat_da.values)
    lon_vals = np.asarray(lon_da.values)
    arr = np.asarray(lai.values)
    if arr.ndim == 2 and lat_vals.ndim == 1 and lon_vals.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        return pd.DataFrame({"LAT": lat2d.ravel(), "LON": lon2d.ravel(), "LAI": arr.ravel()})
    if arr.ndim == 1 and lat_vals.ndim == 1 and lon_vals.ndim == 1 and \
       arr.size == lat_vals.size == lon_vals.size:
        return pd.DataFrame({"LAT": lat_vals.ravel(), "LON": lon_vals.ravel(), "LAI": arr.ravel()})
    # Attempt broadcast fallback
    try:
        lat_b, lon_b = xr.broadcast(xr.DataArray(lat_da), xr.DataArray(lon_da))
        return pd.DataFrame({
            "LAT": np.asarray(lat_b.values).ravel(),
            "LON": np.asarray(lon_b.values).ravel(),
            "LAI": arr.ravel()
        })
    except Exception:
        return pd.DataFrame({"LAT": [], "LON": [], "LAI": []})

# ── SOC aggregations ─────────────────────────────────────────────────────────
def soc_past_monthly_region_means(ds_past: xr.Dataset, year: int, soc_map: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-month regional means for a PAST year (NetCDF grid).
    Robust to any monthly timestamp (mid-month, month-end, non-Gregorian).
    Returns DataFrame with columns [REGION, date, Total_C].
    """
    da = ds_past["total_C"]  # (time, lat, lon)
    time = da["time"]
    # mask this year
    mask_year = xr.DataArray(time.dt.year == year, dims=time.dims)
    da_year = da.sel(time=mask_year)

    if da_year.sizes.get("time", 0) == 0:
        return pd.DataFrame(columns=[REGION_FIELD, "date", "Total_C"])

    lat_vals = np.asarray(ds_past["lat"].values)
    lon_vals = np.asarray(ds_past["lon"].values)
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

    # Pre-index mapping for fast join
    map_idx = soc_map.set_index(["LAT","LON"])[REGION_FIELD]

    rows = []
    # loop actual months present in this year
    months_present = np.unique(da_year["time"].dt.month.values)
    for mo in months_present:
        # select all timesteps of this (year, month), then mean across time (usually 1 step)
        mask_month = xr.DataArray(da_year["time"].dt.month == mo, dims=da_year["time"].dims)
        da_m = da_year.sel(time=mask_month).mean(dim="time", skipna=True)

        arr = np.asarray(da_m.values)
        df = pd.DataFrame({
            "LAT":  lat2d.ravel(),
            "LON":  lon2d.ravel(),
            "Total_C": arr.ravel()
        })
        # join to regions and average spatially per region
        df = df.join(map_idx, on=["LAT","LON"], how="inner").dropna(subset=["Total_C", REGION_FIELD])
        if df.empty:
            continue

        m = df.groupby(REGION_FIELD, as_index=False)["Total_C"].mean()
        # Use a canonical day=1 date for the label
        m["date"] = pd.Timestamp(year=year, month=int(mo), day=1)
        rows.append(m)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[REGION_FIELD, "date", "Total_C"])


def soc_parquet_monthly_region_means(dir_path: Path, year: int, soc_map: pd.DataFrame) -> pd.DataFrame:
    """Compute per-month regional means for a parquet-based year (Present/Future)."""
    map_idx = soc_map.set_index(["LAT","LON"])[REGION_FIELD]
    rows = []
    for mo in range(1, 13):
        fp = dir_path / f"SOC_terms_{year}_{mo:02d}_River.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp, columns=["LAT","LON","Total_C"]).dropna(subset=["Total_C"])
        df = df.join(map_idx, on=["LAT","LON"], how="inner")
        if df.empty:
            continue
        m = df.groupby(REGION_FIELD, as_index=False)["Total_C"].mean()
        m["date"] = pd.Timestamp(year=year, month=mo, day=1)
        rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[REGION_FIELD, "date", "Total_C"])

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    regions = load_regions(REGIONS_PATH)
    rminx, rminy, rmaxx, rmaxy = regions.total_bounds
    print(f"[Regions bbox] lon[{rminx:.4f}, {rmaxx:.4f}] lat[{rminy:.4f}, {rmaxy:.4f}]")

    # --- Build mapping for SOC past grid (from NetCDF lat/lon) ---
    if not PAST_SOC_NC.exists():
        raise FileNotFoundError(f"Past SOC netCDF missing: {PAST_SOC_NC}")
    ds_past = xr.open_dataset(PAST_SOC_NC)
    past_lat, past_lon = np.asarray(ds_past["lat"].values), np.asarray(ds_past["lon"].values)
    soc_map_past = build_point_region_map_from_xy(past_lat, past_lon, regions, tag="SOC_PAST")

    # --- Build mapping for SOC parquet grid (from one present file; reused for future too) ---
    sample_present = find_one_parquet(PRESENT_SOC_DIR, YEARS_PRESENT)
    soc_map_parquet = build_point_region_map_from_parquet(sample_present, regions, tag="SOC_PARQUET")

    # --- Build LAI mappings & open datasets ---
    # Past LAI
    ds_lai_past    = open_lai_dataset(LAI_PAST_FP)
    lai_past_lat, lai_past_lon = extract_lai_latlon(ds_lai_past)
    lai_map_past   = build_point_region_map_from_xy(lai_past_lat, lai_past_lon, regions, tag="LAI_PAST")

    # Present LAI (2001–2014)
    ds_lai_present = open_lai_dataset(LAI_PRESENT_FP)
    lai_pres_lat, lai_pres_lon = extract_lai_latlon(ds_lai_present)
    lai_map_present = build_point_region_map_from_xy(lai_pres_lat, lai_pres_lon, regions, tag="LAI_PRESENT")

    # Future LAI per SSP (2015–2100). If missing, fall back to 245 file.
    lai_ds_by_ssp = {}
    lai_map_by_ssp = {}
    for ssp in SSPS:
        path = FUTURE_SSP_LAI.get(ssp, None)
        ds = open_lai_dataset(path, fallback=LAI_2015_245_FP)
        lai_ds_by_ssp[ssp] = ds
        lat, lon = extract_lai_latlon(ds)
        lai_map_by_ssp[ssp] = build_point_region_map_from_xy(lat, lon, regions, tag=f"LAI_ssp{ssp}")

    # Also open the 2015–2100_245 file explicitly for 2015–2024 (present years per requirement)
    ds_lai_2015_245 = open_lai_dataset(LAI_2015_245_FP)

    records = []

    try:
        # ===== Past (1950–2006) =====
        for year in YEARS_PAST:
            print(f"\n[Past] {year}")
            # SOC monthly → annual stats
            monthly = soc_past_monthly_region_means(ds_past, year, soc_map_past)
            if monthly.empty:
                print("  - No SOC monthly records for this year.")
                continue
            soc_annual = (
                monthly.groupby(REGION_FIELD, as_index=False)["Total_C"]
                       .agg(soc_mean="mean", soc_std="std")
            )

            # LAI annual mean per region (1950–2000 from LAI_PAST, 2001–2006 from LAI_PRESENT)
            if year <= 2000:
                lai_df = lai_annual_df(ds_lai_past, year)
                lai_map = lai_map_past
            else:  # 2001–2006
                lai_df = lai_annual_df(ds_lai_present, year)
                lai_map = lai_map_present

            if lai_df.empty:
                lai_reg = pd.DataFrame({REGION_FIELD: [], "lai_mean": []})
            else:
                lai_reg = (
                    lai_df.merge(lai_map, on=["LAT","LON"], how="inner")
                          .dropna(subset=["LAI"])
                          .groupby(REGION_FIELD, as_index=False)["LAI"]
                          .mean()
                          .rename(columns={"LAI": "lai_mean"})
                )

            merged = soc_annual.merge(lai_reg, on=REGION_FIELD, how="left")
            merged["year"] = year
            merged["period"] = "Past"
            merged["scenario"] = "Past"
            records.append(merged)

        # ===== Present (2007–2024) =====
        for year in YEARS_PRESENT:
            print(f"\n[Present] {year}")
            # SOC monthly → annual stats (parquet grid)
            monthly = soc_parquet_monthly_region_means(PRESENT_SOC_DIR, year, soc_map_parquet)
            if monthly.empty:
                print("  - No SOC monthly records for this year.")
                continue
            soc_annual = (
                monthly.groupby(REGION_FIELD, as_index=False)["Total_C"]
                       .agg(soc_mean="mean", soc_std="std")
            )

            # LAI: 2007–2014 from present file; 2015–2024 forced to 245 file (per your requirement)
            if year <= 2014:
                lai_df = lai_annual_df(ds_lai_present, year)
                lai_map = lai_map_present
            else:
                lai_df = lai_annual_df(ds_lai_2015_245, year)
                lai_map = lai_map_by_ssp["245"]

            if lai_df.empty:
                lai_reg = pd.DataFrame({REGION_FIELD: [], "lai_mean": []})
            else:
                lai_reg = (
                    lai_df.merge(lai_map, on=["LAT","LON"], how="inner")
                          .dropna(subset=["LAI"])
                          .groupby(REGION_FIELD, as_index=False)["LAI"]
                          .mean()
                          .rename(columns={"LAI": "lai_mean"})
                )

            merged = soc_annual.merge(lai_reg, on=REGION_FIELD, how="left")
            merged["year"] = year
            merged["period"] = "Present"
            merged["scenario"] = "Present"
            records.append(merged)

        # ===== Future (2025–2100) by SSP =====
        for ssp in SSPS:
            print(f"\n[Future ssp{ssp}]")
            ssp_dir = FUTURE_SOC_DIR / ssp
            if not ssp_dir.exists():
                warnings.warn(f"SOC future dir missing for ssp{ssp}: {ssp_dir}")
                continue
            lai_ds = lai_ds_by_ssp[ssp]
            lai_map = lai_map_by_ssp[ssp]

            for year in YEARS_FUTURE:
                # SOC monthly → annual stats (parquet grid; reuse same mapping)
                monthly = soc_parquet_monthly_region_means(ssp_dir, year, soc_map_parquet)
                if monthly.empty:
                    # Some SSP directories may start later; skip silently
                    continue
                soc_annual = (
                    monthly.groupby(REGION_FIELD, as_index=False)["Total_C"]
                           .agg(soc_mean="mean", soc_std="std")
                )

                # LAI annual mean per region for this SSP/year
                lai_df = lai_annual_df(lai_ds, year)
                if lai_df.empty:
                    lai_reg = pd.DataFrame({REGION_FIELD: [], "lai_mean": []})
                else:
                    lai_reg = (
                        lai_df.merge(lai_map, on=["LAT","LON"], how="inner")
                              .dropna(subset=["LAI"])
                              .groupby(REGION_FIELD, as_index=False)["LAI"]
                              .mean()
                              .rename(columns={"LAI": "lai_mean"})
                    )

                merged = soc_annual.merge(lai_reg, on=REGION_FIELD, how="left")
                merged["year"] = year
                merged["period"] = "Future"
                merged["scenario"] = f"ssp{ssp}"
                records.append(merged)

        # ===== Write output =====
        out = pd.concat(records, ignore_index=True)
        out = out[["year","period","scenario",REGION_FIELD,"soc_mean","soc_std","lai_mean"]] \
                 .sort_values(["year","scenario",REGION_FIELD]).reset_index(drop=True)

        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_CSV, index=False)
        print(f"\n✔ Saved per-year per-region SOC/LAI summary → {OUT_CSV}")

    finally:
        # Close open datasets
        ds_past.close()
        ds_lai_past.close()
        ds_lai_present.close()
        ds_lai_2015_245.close()
        for ds in lai_ds_by_ssp.values():
            ds.close()

if __name__ == "__main__":
    main()
