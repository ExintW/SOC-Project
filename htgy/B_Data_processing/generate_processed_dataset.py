import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # noqa: E402,F401


BOUNDARY_SHAPEFILE = DATA_DIR / "Loess_Plateau_vector_border.shp"
POINTS_CSV = DATA_DIR / "Loess_Plateau_Points.csv"
DEM_TIF = DATA_DIR / "htgyDEM.tif"
DAM_SOURCES = {
    "Backbone Dams": DATA_DIR / "骨干坝" / "骨干坝.shp",
    "Medium-sized Dams": DATA_DIR / "中型坝" / "中型坝.shp",
}
K1_TIF = DATA_DIR / "k1_halfDegree.tif"
K2_TIF = DATA_DIR / "k2_halfDegree.tif"
FINAL_GRID_CSV = PROCESSED_DIR / "Resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv"
DAM_MATCHES_CSV = PROCESSED_DIR / "htgy_Dam_with_matched_points.csv"


def ensure_crs(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError(
            f"{layer_name} is missing a CRS. Please ensure the corresponding .prj file is present."
        )
    return gdf


def build_resampled_grid(boundary_wgs84: gpd.GeoDataFrame) -> pd.DataFrame:
    minx, miny, maxx, maxy = boundary_wgs84.total_bounds
    grid_res_lat = 0.00898
    grid_res_lon = 0.01084
    x_coords = np.arange(minx, maxx + grid_res_lon, grid_res_lon)
    y_coords = np.arange(miny, maxy + grid_res_lat, grid_res_lat)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_df = pd.DataFrame({"LON": xx.ravel(), "LAT": yy.ravel()})

    sample_df = pd.read_csv(POINTS_CSV, encoding="ISO-8859-1")
    sample_df["LON"] = pd.to_numeric(sample_df["LON"], errors="coerce")
    sample_df["LAT"] = pd.to_numeric(sample_df["LAT"], errors="coerce")
    sample_df = sample_df.dropna(subset=["LON", "LAT"])

    points = sample_df[["LON", "LAT"]].values
    tree = cKDTree(points)

    numeric_columns = [
        col
        for col in sample_df.select_dtypes(include=[np.number]).columns
        if col not in {"LON", "LAT"}
    ]

    for col in numeric_columns:
        values = sample_df[col].values
        interp_linear = griddata(points, values, (xx, yy), method="linear")
        interp_nearest = griddata(points, values, (xx, yy), method="nearest")
        interp_combined = np.where(np.isnan(interp_linear), interp_nearest, interp_linear)
        grid_df[col] = interp_combined.ravel()

    if "ORGA" in grid_df.columns:
        grid_df["ORGA"] = grid_df["ORGA"] * 10 * 0.58

    if "LANDUSE" in sample_df.columns:
        _, nearest_idx = tree.query(grid_df[["LON", "LAT"]].values, k=1)
        grid_df["LANDUSE"] = sample_df["LANDUSE"].iloc[nearest_idx].values

    grid_gdf = gpd.GeoDataFrame(
        grid_df,
        geometry=gpd.points_from_xy(grid_df["LON"], grid_df["LAT"]),
        crs="EPSG:4326",
    )
    boundary_union = boundary_wgs84.unary_union
    grid_gdf = grid_gdf[grid_gdf.geometry.within(boundary_union)]
    grid_gdf = grid_gdf.reset_index(drop=True)
    return pd.DataFrame(grid_gdf.drop(columns="geometry"))


def add_dem(grid_df: pd.DataFrame, boundary_wgs84: gpd.GeoDataFrame) -> pd.DataFrame:
    points_gdf = gpd.GeoDataFrame(
        grid_df.copy(),
        geometry=gpd.points_from_xy(grid_df["LON"], grid_df["LAT"]),
        crs="EPSG:4326",
    )
    points_gdf["__orig_idx"] = np.arange(len(points_gdf))

    with rasterio.open(DEM_TIF) as dem_dataset:
        dem_crs = dem_dataset.crs
        boundary_dem = boundary_wgs84.to_crs(dem_crs)
        points_in_dem = points_gdf.to_crs(dem_crs)

        points_within_boundary = gpd.sjoin(
            points_in_dem,
            boundary_dem,
            how="inner",
            predicate="within",
        )
        if points_within_boundary.empty:
            raise ValueError("No grid points fall inside the Loess Plateau boundary in DEM CRS.")
        points_within_boundary = points_within_boundary.sort_values("__orig_idx")

        coords = [(geom.x, geom.y) for geom in points_within_boundary.geometry]
        samples = list(dem_dataset.sample(coords))
        dem_values = np.array([sample[0] if sample.size > 0 else np.nan for sample in samples])

    if np.isnan(dem_values).any():
        valid_mask = ~np.isnan(dem_values)
        if valid_mask.any():
            valid_coords = np.column_stack(
                (
                    points_within_boundary.geometry.x[valid_mask],
                    points_within_boundary.geometry.y[valid_mask],
                )
            )
            tree = cKDTree(valid_coords)
            missing_coords = np.column_stack(
                (
                    points_within_boundary.geometry.x[~valid_mask],
                    points_within_boundary.geometry.y[~valid_mask],
                )
            )
            _, idx = tree.query(missing_coords, k=1)
            dem_values[~valid_mask] = dem_values[valid_mask][idx]
        else:
            raise ValueError("DEM sampling failed for all grid points.")

    points_within_boundary["htgy_DEM"] = dem_values
    points_within_boundary = points_within_boundary.sort_values("__orig_idx")
    result_df = points_within_boundary.drop(columns=["geometry"]).set_index("__orig_idx")
    result_df = result_df.sort_index()
    return result_df.reset_index(drop=True)


def load_dams() -> pd.DataFrame:
    dam_frames = []
    for label, path in DAM_SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing dam shapefile: {path}")
        gdf = gpd.read_file(path)
        if gdf.empty:
            continue
        gdf = ensure_crs(gdf, f"{label} shapefile")
        gdf = gdf.to_crs("EPSG:4326")
        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y
        gdf["dam_type"] = label
        dam_frames.append(gdf.drop(columns="geometry"))

    if not dam_frames:
        raise ValueError("No dam records were found in the supplied shapefiles.")

    merged = pd.concat(dam_frames, ignore_index=True)
    return merged


def label_grid_with_dams(grid_df: pd.DataFrame, dam_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grid_df = grid_df.copy().reset_index(drop=True)
    dam_df = dam_df.copy().reset_index(drop=True)

    required_dam_cols = {"x", "y"}
    if not required_dam_cols.issubset(dam_df.columns):
        missing = ", ".join(sorted(required_dam_cols - set(dam_df.columns)))
        raise ValueError(f"Dam data is missing required coordinate columns: {missing}")

    grid_points = grid_df[["LON", "LAT"]].to_numpy()
    dam_points = dam_df[["x", "y"]].to_numpy()

    tree = cKDTree(grid_points)
    _, nearest_indices = tree.query(dam_points)

    matched_dams = dam_df.copy()
    matched_dams["matched_Lon"] = grid_df.loc[nearest_indices, "LON"].values
    matched_dams["matched_Lat"] = grid_df.loc[nearest_indices, "LAT"].values

    grid_df["Region"] = "erosion area"
    matched_index_set = set(nearest_indices.tolist())
    grid_df.loc[grid_df.index.isin(matched_index_set), "Region"] = "sedimentation area"

    return grid_df, matched_dams


def interpolate_tiff(path: Path, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    data_array = rioxarray.open_rasterio(path)
    nodata = data_array.rio.nodata
    data_array = data_array.where((data_array != nodata) & (data_array >= 0.0))
    data_array = data_array.squeeze("band", drop=True).rename({"x": "lon", "y": "lat"})

    lon_da = xr.DataArray(lons, dims="points")
    lat_da = xr.DataArray(lats, dims="points")

    linear = data_array.interp(lon=lon_da, lat=lat_da, method="linear")
    nearest = data_array.interp(lon=lon_da, lat=lat_da, method="nearest")
    return linear.fillna(nearest).values


def convert_som_to_soc_monthly(k_day: np.ndarray) -> np.ndarray:
    k_day = np.maximum(k_day, 0.0)
    return (1 - np.exp(-k_day * 30)) * 0.58


def add_k_parameters(grid_df: pd.DataFrame) -> pd.DataFrame:
    df = grid_df.copy()
    lon_values = df["LON"].values
    lat_values = df["LAT"].values

    som_k1_day = interpolate_tiff(K1_TIF, lon_values, lat_values)
    som_k2_day = interpolate_tiff(K2_TIF, lon_values, lat_values)

    df["SOC_k1_fast_pool (1/day)"] = som_k1_day * 0.58
    df["SOC_k2_slow_pool (1/day)"] = som_k2_day * 0.58
    df["SOC_k1_fast_pool (1/month)"] = convert_som_to_soc_monthly(som_k1_day)
    df["SOC_k2_slow_pool (1/month)"] = convert_som_to_soc_monthly(som_k2_day)

    for column in ["SOC_k1_fast_pool (1/month)", "SOC_k2_slow_pool (1/month)"]:
        grid = df.pivot(index="LAT", columns="LON", values=column)
        grid = grid.sort_index(ascending=True).reindex(sorted(grid.columns), axis=1)
        lons_mesh, lats_mesh = np.meshgrid(grid.columns.values, grid.index.values)

        values = grid.values
        points = np.column_stack((lons_mesh.ravel(), lats_mesh.ravel()))
        flat_values = values.ravel()
        valid_mask = ~np.isnan(flat_values)

        if valid_mask.any():
            filled = griddata(points[valid_mask], flat_values[valid_mask], points, method="nearest")
            filled_matrix = filled.reshape(values.shape)
        else:
            filled_matrix = np.zeros_like(values)

        smoothed = gaussian_filter(filled_matrix, sigma=20.0)
        smooth_df = pd.DataFrame(smoothed, index=grid.index, columns=grid.columns)
        df[column] = df.apply(lambda row: smooth_df.at[row["LAT"], row["LON"]], axis=1)

    return df


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Starting processed dataset generation pipeline...")
    print("Loading Loess Plateau boundary shapefile...")
    boundary = ensure_crs(gpd.read_file(BOUNDARY_SHAPEFILE), "Loess Plateau boundary")
    boundary_wgs84 = boundary.to_crs("EPSG:4326")

    print("Building 1 km grid and interpolating soil attributes...")
    grid_df = build_resampled_grid(boundary_wgs84)
    print(f"Grid contains {len(grid_df)} cells after clipping to boundary.")

    print("Sampling DEM values for grid cells...")
    grid_with_dem = add_dem(grid_df, boundary_wgs84)

    print("Loading dam shapefiles and preparing dam dataset...")
    dam_df = load_dams()
    print(f"Loaded {len(dam_df)} dam records.")

    print("Matching dams to grid cells and labeling regions...")
    grid_with_region, dam_matches = label_grid_with_dams(grid_with_dem, dam_df)

    print("Interpolating k1/k2 parameters and applying smoothing...")
    final_df = add_k_parameters(grid_with_region)

    final_df.to_csv(FINAL_GRID_CSV, index=False, encoding="utf-8-sig")
    dam_matches.to_csv(DAM_MATCHES_CSV, index=False, encoding="utf-8-sig")

    print(f"Final dataset saved to {FINAL_GRID_CSV}")
    print(f"Dam match table saved to {DAM_MATCHES_CSV}")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()

