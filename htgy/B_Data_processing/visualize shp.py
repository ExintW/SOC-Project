#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Merge subregions -> regions, fix CRS/geometry, compute metric areas, print ALL rows,
# visualize, and SAVE (GPKG authoritative, SHP lean, optional GeoJSON).

import sys, os
import pathlib, warnings
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# project globals
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import DATA_DIR, OUTPUT_DIR

# ====== CONFIG ======
SHP_PATH  = DATA_DIR / "Ecological regionalization" / "Ecological_regionalization.shp"
OUT_DIR   = DATA_DIR / "Ecological regionalization" / "Processed"
SAVE_PNG  = True
MAKE_FOLIUM = False        # set True if folium installed
SAVE_SHP    = True         # export a lean SHP (short fields)
SAVE_GPKG   = True         # authoritative output (keeps fields/precision)
SAVE_GEOJSON= False        # optional
FORCE_CRS_EPSG4326 = True  # source coords are degrees but mislabeled -> override
# =====================

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pandas print all rows/cols
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
pd.set_option("display.max_colwidth", None)

src = pathlib.Path(SHP_PATH)
if not src.exists():
    raise FileNotFoundError(f"Shapefile not found: {src}")

gdf = gpd.read_file(src)

# --- Map subcodes to regions ---
def to_region(code: str) -> str:
    c = str(code).strip().upper() if code is not None else "UNKNOWN"
    if c in ("A1", "A2"): return "A"
    if c in ("B1", "B2"): return "B"
    return c  # keep C, D, etc.

if "CODE" not in gdf.columns:
    raise KeyError("Expected a 'CODE' column (A1, A2, B1, B2, C, D).")
gdf["REGION"] = gdf["CODE"].apply(to_region)

# --- Fix CRS: layer is in degrees; override to EPSG:4326 (SET, don't transform) ---
def looks_like_degrees(g: gpd.GeoDataFrame) -> bool:
    minx, miny, maxx, maxy = g.total_bounds
    return (abs(minx) <= 180) and (abs(maxx) <= 180) and (abs(miny) <= 90) and (abs(maxy) <= 90)

if FORCE_CRS_EPSG4326:
    if gdf.crs is None:
        if looks_like_degrees(gdf):
            warnings.warn("Regions have no CRS; treating as EPSG:4326.")
            gdf = gdf.set_crs(4326, allow_override=True)
        else:
            raise ValueError("Regions have no CRS and do not look like degrees; confirm CRS.")
    else:
        epsg = gdf.crs.to_epsg()
        if epsg != 4326 and looks_like_degrees(gdf):
            warnings.warn(f"Regions report CRS {gdf.crs}, but bounds look like degrees. "
                          "Overriding to EPSG:4326 (no transform).")
            gdf = gdf.set_crs(4326, allow_override=True)
        elif epsg != 4326:
            gdf = gdf.to_crs(4326)

# --- Geometry repair BEFORE dissolve ---
def make_valid_series(geom_series: gpd.GeoSeries) -> gpd.GeoSeries:
    try:
        from shapely.validation import make_valid  # Shapely 2.x
        return geom_series.apply(lambda g: make_valid(g) if g is not None else g)
    except Exception:
        # Shapely 1.x fallback
        return geom_series.buffer(0)

gdf["geometry"] = make_valid_series(gdf.geometry)
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
gdf = gdf.explode(index_parts=False)

# --- Dissolve by REGION ---
# (no need to sum attributes; we keep REGION + geometry and recompute metrics later)
diss = gdf.dissolve(by="REGION", as_index=False)

# --- Geometry repair AFTER dissolve ---
diss["geometry"] = make_valid_series(diss.geometry)
diss = diss[~diss.geometry.is_empty & diss.geometry.notna()].copy()
diss = diss.explode(index_parts=False)
diss = diss.dissolve(by="REGION", as_index=False)

# --- Compute proper metric areas/lengths in equal-area CRS ---
# Use EPSG:6933 (World Cylindrical Equal Area) for robust areas & perimeters in meters
try:
    d_eq = diss.to_crs(6933)
    diss["AREA_m2"]  = d_eq.geometry.area
    diss["PERIM_m"]  = d_eq.geometry.length
except Exception as e:
    warnings.warn(f"Could not compute area/length in EPSG:6933: {e}")
    # Fallback to 3395 (meters) if needed
    d3395 = diss.to_crs(3395)
    diss["AREA_m2"]  = d3395.geometry.area
    diss["PERIM_m"]  = d3395.geometry.length

# Scaled, human-friendly versions (avoid huge numbers in SHP DBF)
diss["AREA_km2"] = (diss["AREA_m2"] / 1e6).round(2)
diss["PERIM_km"] = (diss["PERIM_m"] / 1e3).round(2)

# --- Column order: REGION first, keep useful metrics before geometry ---
first_cols = ["REGION", "AREA_km2", "PERIM_km", "AREA_m2", "PERIM_m"]
other_cols = [c for c in diss.columns if c not in first_cols + ["geometry"]]
diss = diss[first_cols + other_cols + ["geometry"]]

# --- Print ALL rows (no geometry) ---
print("\n=== DISSOLVED BY REGION (ALL ROWS, geometry excluded) ===")
attrs = diss.drop(columns="geometry", errors="ignore")
print(attrs.to_string(index=False))

# --- Save outputs ---
base_name = src.stem + "_REGIONS_clean"

# Authoritative GeoPackage (no field-width limits)
if SAVE_GPKG:
    gpkg_out = OUT_DIR / f"{base_name}.gpkg"
    diss.to_file(gpkg_out, layer="regions", driver="GPKG")
    print(f"✔ Saved GeoPackage → {gpkg_out}")

# Lean Shapefile (keep only compact fields so DBF doesn't overflow)
if SAVE_SHP:
    shp_out = OUT_DIR / f"{base_name}.shp"
    shp_cols = ["REGION", "AREA_km2", "PERIM_km", "geometry"]
    diss_shp = diss[shp_cols].copy()
    # cast to float32 to reduce DBF width; SHP will still truncate names to ≤10 chars
    diss_shp["AREA_km2"] = diss_shp["AREA_km2"].astype("float32")
    diss_shp["PERIM_km"] = diss_shp["PERIM_km"].astype("float32")
    diss_shp.to_file(shp_out, driver="ESRI Shapefile", encoding="utf-8")
    print(f"✔ Saved shapefile → {shp_out}")

if SAVE_GEOJSON:
    geojson_out = OUT_DIR / f"{base_name}.geojson"
    diss.to_file(geojson_out, driver="GeoJSON")
    print(f"✔ Saved GeoJSON → {geojson_out}")

# --- Static plot ---
fig, ax = plt.subplots(figsize=(9, 7))
diss.plot(ax=ax, column="REGION", edgecolor="black", linewidth=0.6, legend=True)
ax.set_title("Ecological Regionalization (Cleaned & Dissolved)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_aspect("equal", adjustable="box")
plt.tight_layout()
if SAVE_PNG:
    out_png = OUT_DIR / f"{base_name}_preview.png"
    fig.savefig(out_png, dpi=220)
    print(f"✔ Saved static PNG → {out_png}")
else:
    plt.show()


gdf = gpd.read_file(DATA_DIR / "Ecological regionalization" / "Ecological_regionalization.shp")
print(gdf.columns)

# 看前几行数据，确认每列代表什么
print(gdf.head())


# --- Optional interactive HTML map ---
if MAKE_FOLIUM:
    try:
        import folium
        d_wgs84 = diss  # already EPSG:4326
        minx, miny, maxx, maxy = d_wgs84.total_bounds
        center = [(miny + maxy) / 2, (minx + maxx) / 2]
        m = folium.Map(location=center, zoom_start=6, control_scale=True)
        folium.GeoJson(
            d_wgs84.__geo_interface__,
            name="regions",
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in d_wgs84.columns if c != "geometry"],
                sticky=False, labels=True
            ),
        ).add_to(m)
        folium.LayerControl().add_to(m)
        out_html = OUT_DIR / f"{base_name}_map.html"
        m.save(out_html)
        print(f"✔ Saved interactive HTML → {out_html}")
    except ImportError:
        warnings.warn("Folium not installed. Run: pip install folium")
