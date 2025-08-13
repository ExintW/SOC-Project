import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─── project paths ─────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import DATA_DIR, OUTPUT_DIR

# ─── safe metric function ─────────────────────────────────────────────────────
def calc_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """Compute ME, MAE, RMSE, PBIAS, R2, NSE, KGE with safe-guards."""
    mask = np.isfinite(obs) & np.isfinite(pred)
    o = obs[mask]
    m = pred[mask]

    # need at least two points for correlation-based metrics
    if o.size < 2:
        return {k: np.nan for k in ("ME", "MAE", "RMSE", "PBIAS", "R2", "NSE", "KGE")}

    ME   = np.mean(m - o)
    MAE  = mean_absolute_error(o, m)
    RMSE = np.sqrt(mean_squared_error(o, m))

    sum_o = np.sum(o)
    PBIAS = (np.sum(m - o) / sum_o * 100) if sum_o != 0 else np.nan

    std_o = np.std(o, ddof=1)
    std_m = np.std(m, ddof=1)
    if std_o == 0 or std_m == 0:
        r, R2 = np.nan, np.nan
    else:
        r  = np.corrcoef(o, m)[0,1]
        R2 = r**2

    denom = np.sum((o - o.mean())**2)
    NSE = (1 - np.sum((o - m)**2) / denom) if denom != 0 else np.nan

    if np.isnan(r):
        KGE = np.nan
    else:
        alpha = std_m / std_o
        beta  = np.mean(m) / np.mean(o) if np.mean(o) != 0 else np.nan
        KGE   = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return dict(ME=ME, MAE=MAE, RMSE=RMSE,
                PBIAS=PBIAS, R2=R2, NSE=NSE, KGE=KGE)

# ─── validation workflow ──────────────────────────────────────────────────────
input_xlsx = OUTPUT_DIR / "River_Basin_Points_Filtered.xlsx"
xls        = pd.ExcelFile(input_xlsx)
all_metrics = []

for sheet_name in xls.sheet_names:
    # 1) load observed points
    obs_df = (
        xls.parse(sheet_name)
           .rename(columns={"X":"lon","Y":"lat","SOC (g/kg)":"obs_soc"})
           .dropna(subset=["lon","lat","obs_soc"])
    )

    if obs_df.empty:
        print(f"[SKIP] '{sheet_name}': no obs → skipping basin")
        continue

    year = sheet_name.split("-")[-1]

    # 2) find model files
    model_dir = OUTPUT_DIR / "Data" / "SOC_Present 6"
    files = sorted(model_dir.glob(f"SOC_terms_{year}_*_River.parquet"))
    if not files:
        print(f"[SKIP] '{sheet_name}': no model files for {year}")
        continue

    # 3) read & concat monthly
    monthly = []
    for fp in files:
        dfm = pd.read_parquet(fp).rename(columns={"LAT":"lat","LON":"lon"})
        dfm["soc_model"] = dfm["C_fast"] + dfm["C_slow"]
        monthly.append(dfm[["lat","lon","soc_model"]])
    model_all = pd.concat(monthly, ignore_index=True)

    # 4) annual mean
    model_ann = (
        model_all
        .groupby(["lat","lon"], as_index=False)["soc_model"]
        .mean()
    )

    # 5) interpolate (nearest)
    pts       = model_ann[["lon","lat"]].values
    vals      = model_ann["soc_model"].values
    coords    = obs_df[["lon","lat"]].values
    obs_df["pred"] = griddata(pts, vals, coords, method="nearest")

    # — new check: skip if no variance in obs or pred —
    if np.std(obs_df["obs_soc"], ddof=1) == 0 or np.std(obs_df["pred"], ddof=1) == 0:
        print(f"[SKIP] '{sheet_name}': zero variance → skipping basin")
        continue

    # 6) compute metrics
    mets = calc_metrics(obs_df["obs_soc"].values, obs_df["pred"].values)
    mets["basin"] = sheet_name
    all_metrics.append(mets)

    # 7) save paired obs/pred
    out_dir = OUTPUT_DIR / "Validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_fp = out_dir / f"{sheet_name}_paired.csv"
    obs_df[["lon","lat","obs_soc","pred"]].to_csv(paired_fp, index=False)
    print(f"Saved paired data for {sheet_name} to {paired_fp}")

# 8) write summary
if all_metrics:
    dfm = pd.DataFrame(all_metrics)[
        ["basin","ME","MAE","RMSE","PBIAS","R2","NSE","KGE"]
    ]
    summary_fp = OUTPUT_DIR / "Validation" / "basin_metrics.csv"
    dfm.to_csv(summary_fp, index=False)
    print(f"\nSummary metrics saved to {summary_fp}")
else:
    print("No basins passed the variance check; no summary written.")
