#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Future SOC potential (max monthly) & LAI mean by region for 2025–2100.

Definition:
  - SOC potential (per year, per region) = MAX over the 12 monthly regional SOC means.
  - LAI mean (per year, per region)      = annual mean LAI over that region (as before).

Scenarios: ssp126, ssp245, ssp370, ssp585
Regions:   uses 'REGION' from GeoPackage (EPSG:4326)

Output CSV:
  OUTPUT_DIR / "soc_potential_maxmonth_by_region_2025_2100.csv"

Columns:
  year, period(Future), scenario(sspXXX), REGION, soc_potential, lai_mean
"""

import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

# ── 0) Project globals ────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

# ── 1) Paths & config ─────────────────────────────────────────────────────────
REGIONS_PATH = DATA_DIR / "Ecological regionalization" / "Processed" / "Ecological_regionalization_REGIONS_clean.gpkg"
REGION_FIELD = "REGION"

PRESENT_SOC_DIR = OUTPUT_DIR / "Data" / "SOC_Present 7"
FUTURE_SOC_DIR  = OUTPUT_DIR / "Data" / "SOC_Future 7"  # subdirs: 126,245,370,585

# LAI sources (future by SSP with fallback to 245)
LAI_2015_245_FP = PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_245.nc"
FUTURE_SSP_LAI = {
    "126": PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_126.nc",
    "245": PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_245.nc",
    "370": PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_370.nc",
    "585": PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / "resampled_lai_points_2015-2100_585.nc",
}

OUT_CSV = OUTPUT_DIR / "soc_potential_maxmonth_by_region_2025_2100.csv"

YEARS_FUTURE = range(2025, 2101)   # 2025–2100 inclusive
SSPS = ["126", "245", "370", "585"]

# ── 2) Helpers: regions & mapping ─────────────────────────────────────────────
def load_regions(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Regions file not found: {path}")
    gdf = gpd.read_file(path)
    if REGION_FIELD not in gdf.columns:
        raise KeyError(f"'{REGION_FIELD}' missing in regions")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(4326, allow_override=True) if gdf.crs is None else gdf.to_crs(4326)
    return gdf[[REGION_FIELD, "geometry"]].copy()

def build_point_region_map_from_parquet(sample_fp: Path, regions: gpd.GeoDataFrame, tag: str) -> pd.DataFrame:
    df = pd.read_parquet(sample_fp, columns=["LAT", "LON"])
    pts = df.drop_duplicates().reset_index(drop=True)
    gpts = gpd.GeoDataFrame(
        pts, geometry=gpd.points_from_xy(pts["LON"].values, pts["LAT"].values), crs="EPSG:4326"
    )
    joined = gpd.sjoin(gpts, regions, how="left", predicate="intersects")
    mapping = (
        joined[["LAT", "LON", REGION_FIELD]]
        .groupby(["LAT", "LON"], sort=False).first()
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

# ── 3) LAI helpers ────────────────────────────────────────────────────────────
def open_lai_dataset(path: Path | None, fallback: Path | None = None) -> xr.Dataset:
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
    if arr.ndim == 1 and lat_vals.ndim == 1 and lon_vals.ndim == 1 and arr.size == lat_vals.size == lon_vals.size:
        return pd.DataFrame({"LAT": lat_vals.ravel(), "LON": lon_vals.ravel(), "LAI": arr.ravel()})
    # broadcast fallback
    lat_b, lon_b = xr.broadcast(xr.DataArray(lat_da), xr.DataArray(lon_da))
    return pd.DataFrame({
        "LAT": np.asarray(lat_b.values).ravel(),
        "LON": np.asarray(lon_b.values).ravel(),
        "LAI": arr.ravel()
    })

# ── 4) SOC monthly means from parquet ────────────────────────────────────────
def soc_parquet_monthly_region_means(dir_path: Path, year: int, soc_map: pd.DataFrame) -> pd.DataFrame:
    map_idx = soc_map.set_index(["LAT", "LON"])[REGION_FIELD]
    rows = []
    for mo in range(1, 13):
        fp = dir_path / f"SOC_terms_{year}_{mo:02d}_River.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp, columns=["LAT", "LON", "Total_C"]).dropna(subset=["Total_C"])
        df = df.join(map_idx, on=["LAT", "LON"], how="inner")
        if df.empty:
            continue
        m = df.groupby(REGION_FIELD, as_index=False)["Total_C"].mean()
        m["date"] = pd.Timestamp(year=year, month=mo, day=1)
        rows.append(m)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=[REGION_FIELD, "date", "Total_C"])

# ── 5) Main: compute future SOC potential & LAI mean ─────────────────────────
def main():
    regions = load_regions(REGIONS_PATH)
    rminx, rminy, rmaxx, rmaxy = regions.total_bounds
    print(f"[Regions bbox] lon[{rminx:.4f}, {rmaxx:.4f}] lat[{rminy:.4f}, {rmaxy:.4f}]")

    # Build SOC mapping from any present parquet file (same grid as future parquet)
    sample_present = find_one_parquet(PRESENT_SOC_DIR, range(2007, 2025))
    soc_map_parquet = build_point_region_map_from_parquet(sample_present, regions, tag="SOC_PARQUET")

    # Pre-open LAI datasets and mappings per SSP
    lai_ds_by_ssp, lai_map_by_ssp = {}, {}
    for ssp in SSPS:
        ds = open_lai_dataset(FUTURE_SSP_LAI.get(ssp), fallback=LAI_2015_245_FP)
        lai_ds_by_ssp[ssp] = ds
        lat, lon = extract_lai_latlon(ds)
        # Map LAI grid points to regions
        pts = pd.DataFrame({"LAT": lat.ravel(), "LON": lon.ravel()}) if lat.ndim == 1 and lon.ndim == 1 else None
        if pts is None:
            # Generalized mapping for 2D lat/lon
            lon2d, lat2d = np.meshgrid(lon, lat) if lat.ndim == 1 and lon.ndim == 1 else (lon, lat)
            pts = pd.DataFrame({"LAT": np.ravel(lat2d), "LON": np.ravel(lon2d)})
        gpts = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["LON"], pts["LAT"]), crs="EPSG:4326")
        joined = gpd.sjoin(gpts, regions, how="left", predicate="intersects")
        lai_map_by_ssp[ssp] = (
            joined[["LAT", "LON", REGION_FIELD]]
            .groupby(["LAT", "LON"], sort=False).first()
            .reset_index()
        )
        matched = lai_map_by_ssp[ssp][REGION_FIELD].notna().sum()
        print(f"[LAI_ssp{ssp} mapping] matched {matched}/{len(lai_map_by_ssp[ssp])} points to regions")

    records = []

    try:
        for ssp in SSPS:
            print(f"\n[Future SOC potential ssp{ssp}]")
            ssp_dir = FUTURE_SOC_DIR / ssp
            if not ssp_dir.exists():
                warnings.warn(f"SOC future dir missing for ssp{ssp}: {ssp_dir}")
                continue

            lai_ds = lai_ds_by_ssp[ssp]
            lai_map = lai_map_by_ssp[ssp]

            for year in YEARS_FUTURE:
                # 1) SOC potential: max over monthly regional means
                monthly = soc_parquet_monthly_region_means(ssp_dir, year, soc_map_parquet)
                if monthly.empty:
                    # No data for this (ssp, year); skip
                    continue
                soc_potential = (
                    monthly.groupby(REGION_FIELD, as_index=False)["Total_C"]
                           .max()
                           .rename(columns={"Total_C": "soc_potential"})
                )

                # 2) LAI annual mean per region for this SSP/year
                lai_df = lai_annual_df(lai_ds, year)
                if lai_df.empty:
                    lai_reg = pd.DataFrame({REGION_FIELD: [], "lai_mean": []})
                else:
                    lai_reg = (
                        lai_df.merge(lai_map, on=["LAT", "LON"], how="inner")
                              .dropna(subset=["LAI"])
                              .groupby(REGION_FIELD, as_index=False)["LAI"]
                              .mean()
                              .rename(columns={"LAI": "lai_mean"})
                    )

                # 3) Merge & annotate
                merged = soc_potential.merge(lai_reg, on=REGION_FIELD, how="left")
                merged["year"] = year
                merged["period"] = "Future"
                merged["scenario"] = f"ssp{ssp}"
                records.append(merged)

        if not records:
            print("No records generated — check inputs.")
            return

        out = pd.concat(records, ignore_index=True)
        out = out[["year", "period", "scenario", REGION_FIELD, "soc_potential", "lai_mean"]] \
                 .sort_values(["year", "scenario", REGION_FIELD]).reset_index(drop=True)

        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_CSV, index=False)
        print(f"\n✔ Saved SOC potential summary → {OUT_CSV}")

    finally:
        for ds in lai_ds_by_ssp.values():
            try:
                ds.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()
