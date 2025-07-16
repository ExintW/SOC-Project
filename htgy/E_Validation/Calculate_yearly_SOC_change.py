import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1) Setup paths & imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import OUTPUT_DIR
# ──────────────────────────────────────────────────────────────────────────────

# 2) Parameters for Pg conversion
area_m2      = 640_000 * 1e6           # 640,000 km² → m²
soil_depth_m = 0.20                    # 20 cm layer
bulk_density = 1300                    # kg/m³
soil_mass_kg = area_m2 * soil_depth_m * bulk_density
conv_factor  = soil_mass_kg / 1e15     # (g/kg → Pg) factor

# 3) Locate your monthly Parquet files
SCENARIO = "245"
base_dir = OUTPUT_DIR / "Data" / "SOC_Future 5" / SCENARIO
all_files = list(base_dir.glob("SOC_terms_*_River.parquet"))

# 4) Extract sorted list of years present
years = sorted({int(fp.name.split("_")[2]) for fp in all_files})

# 5) Compute annual mean Total_C (g/kg/month) using METHOD 2
annual_means = []
for yr in years:
    monthly_dfs = []
    for m in range(1, 13):
        fn = f"SOC_terms_{yr}_{m:02d}_River.parquet"
        fp = base_dir / fn
        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp}")
        df = pd.read_parquet(fp)
        df = df[df["Total_C"].notna()]
        monthly_dfs.append(df[["Total_C"]])
    combined = pd.concat(monthly_dfs, ignore_index=True)
    # mean over ALL grid cells across ALL 12 months
    annual_means.append(combined["Total_C"].mean())

annual_means = np.array(annual_means)  # shape: (n_years,)

# 6) Compute year‑over‑year % change
prev_tot = annual_means[:-1]
next_tot = annual_means[1:]
pct_change = np.divide(
    next_tot - prev_tot,
    prev_tot,
    out=np.full_like(next_tot, np.nan),
    where=(prev_tot != 0)
) * 100

# 7) Compute Pg C / year change
#    Δ_conc is in g/kg/month → multiply by conv_factor → Pg/month → ×12 → Pg/year
delta_g_per_kg    = next_tot - prev_tot
delta_pg_per_year = delta_g_per_kg * conv_factor * 12

# 8) Assemble results and save
df = pd.DataFrame({
    "year":               years[:-1],
    "total_pct_change":   pct_change,
    "total_Pg_per_year":  delta_pg_per_year
})

output_fp = OUTPUT_DIR / f"soc_total_yearly_change_by_method2.csv"
df.to_csv(output_fp, index=False)
print(f"Saved total‐C yearly change to: {output_fp}")
