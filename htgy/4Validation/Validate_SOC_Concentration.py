import os
import sys
from pathlib import Path
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Function to calculate error and efficiency metrics
def calc_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """Compute ME, MAE, RMSE, PBIAS, R2, NSE, KGE between observed and predicted arrays."""
    mask = np.isfinite(obs) & np.isfinite(pred)
    o = obs[mask];
    m = pred[mask]
    if o.size == 0:
        return {k: np.nan for k in ("ME", "MAE", "RMSE", "PBIAS", "R2", "NSE", "KGE")}
    ME = np.mean(m - o)
    MAE = mean_absolute_error(o, m)
    RMSE = np.sqrt(mean_squared_error(o, m))
    PBIAS = np.sum(m - o) / np.sum(o) * 100
    r = np.corrcoef(o, m)[0, 1]
    R2 = r ** 2
    NSE = 1 - np.sum((o - m) ** 2) / np.sum((o - o.mean()) ** 2)
    alpha = np.std(m, ddof=1) / np.std(o, ddof=1)
    beta = np.mean(m) / np.mean(o)
    KGE = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return dict(ME=ME, MAE=MAE, RMSE=RMSE, PBIAS=PBIAS, R2=R2, NSE=NSE, KGE=KGE)


# Read all basin sheets from the Excel file
excel_path = DATA_DIR / "River_Basin_Points.xlsx"
xls = pd.ExcelFile(excel_path)

all_metrics = []

for sheet_name in xls.sheet_names:
    # 1) Load observed points for this basin
    obs_df = xls.parse(sheet_name)
    obs_df = obs_df.rename(columns={
        "X": "lon", "Y": "lat", "SOC (g/kg)": "obs_soc"
    }).dropna(subset=["lon", "lat", "obs_soc"])

    # Extract year from sheet name suffix
    year = sheet_name.split("-")[-1]

    # 2) Find all monthly Parquet files for this year
    model_dir = OUTPUT_DIR / "Data"
    pattern = f"SOC_terms_{year}_*_River.parquet"
    parquet_files = sorted(model_dir.glob(pattern))
    if not parquet_files:
        print(f"[WARN] No model files found for {year}, skipping {sheet_name}")
        continue

    # 3) Read each month, compute soc_model = C_fast + C_slow
    monthly_dfs = []
    for fp in parquet_files:
        df = pd.read_parquet(fp)
        df = df.rename(columns={"LAT": "lat", "LON": "lon"})
        df["soc_model"] = df["C_fast"] + df["C_slow"]
        monthly_dfs.append(df[["lat", "lon", "soc_model"]])
    model_all = pd.concat(monthly_dfs, ignore_index=True)

    # 4) Compute annual mean at each model grid point
    model_ann = (
        model_all
        .groupby(["lat", "lon"], as_index=False)["soc_model"]
        .mean()
    )

    # 5) Interpolate model values to observed point coordinates
    points = model_ann[["lon", "lat"]].values
    values = model_ann["soc_model"].values
    obs_coords = obs_df[["lon", "lat"]].values

    obs_df["pred_nn"] = griddata(points, values, obs_coords, method="nearest")
    obs_df["pred_linear"] = griddata(points, values, obs_coords, method="linear")
    # choose nearest‐neighbor as primary prediction
    obs_df["pred"] = obs_df["pred_nn"]

    # 6) Calculate metrics for this basin
    mets = calc_metrics(obs_df["obs_soc"].values, obs_df["pred"].values)
    mets["basin"] = sheet_name
    all_metrics.append(mets)

    # 7) Save paired obs/pred for this basin
    paired_out = PROCESSED_DIR / f"{sheet_name}_paired.csv"
    obs_df[["lon", "lat", "obs_soc", "pred"]].to_csv(paired_out, index=False)
    print(f"Saved paired data for {sheet_name} to {paired_out}")

# 8) Save summary metrics for all basins
metrics_df = pd.DataFrame(all_metrics)[
    ["basin", "ME", "MAE", "RMSE", "PBIAS", "R2", "NSE", "KGE"]
]
summary_out = PROCESSED_DIR / "basin_metrics.csv"
metrics_df.to_csv(summary_out, index=False)
print(f"\nSummary metrics saved to {summary_out}")
