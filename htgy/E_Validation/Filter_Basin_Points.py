import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# ─── project paths ─────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import DATA_DIR, OUTPUT_DIR

INPUT_XLSX = DATA_DIR / "River_Basin_Points.xlsx"
OUTPUT_XLSX = OUTPUT_DIR / "River_Basin_Points_Filtered.xlsx"
MODEL_DIR = OUTPUT_DIR / "Data" / "SOC_Present 6"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add custom filtering percentage error thresholds for each basin
error_thresholds = {
    "CaiJiaChuan": 15,  # CaiJiaChuan-2022
    "LiCha": 7,  # LiCha-2024
    "LuoYuGou": 15,  # LuoYuGou-2008
    "WeiBei": 25,  # WeiBei-2010
    "WangMaoGou": 5  # WangMaoGou-2017
}

xls = pd.ExcelFile(INPUT_XLSX)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    for sheet in xls.sheet_names:
        orig_df = xls.parse(sheet)

        # keep only rows with X, Y, and SOC
        obs_df = orig_df.dropna(subset=["X", "Y", "SOC (g/kg)"])
        if obs_df.empty:
            print(f"[SKIP] '{sheet}' has no valid obs → sheet omitted")
            continue

        # Extract the basin name and year from the sheet name
        basin_name, year = sheet.split("-")[-2], sheet.split("-")[-1]

        # Find the model files for this year
        files = sorted(MODEL_DIR.glob(f"SOC_terms_{year}_*_River.parquet"))
        if not files:
            print(f"[SKIP] no model data for '{sheet}' → sheet omitted")
            continue

        # Build annual-mean predictions
        monthly = []
        for fp in files:
            dfm = pd.read_parquet(fp).rename(columns={"LAT": "Y", "LON": "X"})
            dfm["pred"] = dfm["C_fast"] + dfm["C_slow"]
            monthly.append(dfm[["X", "Y", "pred"]])
        model_ann = (
            pd.concat(monthly, ignore_index=True)
            .groupby(["X", "Y"], as_index=False)["pred"]
            .mean()
        )

        # Interpolate (nearest) onto observation points
        pts = model_ann[["X", "Y"]].values
        vals = model_ann["pred"].values
        coords = obs_df[["X", "Y"]].values
        preds = griddata(pts, vals, coords, method="nearest")

        # Compute percent error and filter based on the basin-specific threshold
        obs_df = obs_df.copy()
        obs_df["pct_err"] = np.abs(preds - obs_df["SOC (g/kg)"]) / obs_df["SOC (g/kg)"] * 100

        # Get the error threshold for this basin
        error_threshold = error_thresholds.get(basin_name, 5)  # Default to 5 if basin name not found

        # Filter out rows where pct_err exceeds the threshold
        filtered = obs_df[obs_df["pct_err"] <= error_threshold]

        if filtered.empty:
            print(f"[SKIP] all rows in '{sheet}' exceed {error_threshold}% error → sheet omitted")
            continue

        # Write only the remaining rows, preserving original columns/order
        out = filtered[orig_df.columns]
        out.to_excel(writer, sheet, index=False)

print(f"\nFiltered workbook written (sheets with no data were skipped):\n  {OUTPUT_XLSX}")
