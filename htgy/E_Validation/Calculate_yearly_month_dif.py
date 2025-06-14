import os
import sys
from pathlib import Path
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from globals import OUTPUT_DIR  # Assumes dynamic_data.npz is in OUTPUT_DIR

# 1) Load cfast and cslow from dynamic_data.npz
dynamic_npz_path = OUTPUT_DIR / "dynamic_data.npz"
data = np.load(dynamic_npz_path)
cfast = data["soc_fast"]   # shape: (1128, 844, 1263)
cslow = data["soc_slow"]   # shape: (1128, 844, 1263)

# 2) Compute the spatial (grid) mean for each month (ignore NaNs)
mean_fast = np.nanmean(cfast, axis=(1, 2))  # shape: (1128,)
mean_slow = np.nanmean(cslow, axis=(1, 2))  # shape: (1128,)

# 3) Reshape into (years, 12) where year=2007 is row 0
start_year = 2007
months_per_year = 12
n_months = mean_fast.shape[0]                # 1128
n_years = n_months // months_per_year        # 94 years (2007–2100)
fast_matrix = mean_fast.reshape(n_years, months_per_year)
slow_matrix = mean_slow.reshape(n_years, months_per_year)

# 4) Compute year-over-year percentage change for each month:
#    pct_fast[y, m] = 100 * (fast_matrix[y+1, m] − fast_matrix[y, m]) / fast_matrix[y, m]
#    Likewise for pct_slow
#    We use np.divide with where= to avoid division-by-zero warnings;
#    any zero‐denominator yields NaN.

prev_fast = fast_matrix[:-1, :]       # data for years 2007–2099
next_fast = fast_matrix[1:, :]        # data for years 2008–2100

prev_slow = slow_matrix[:-1, :]
next_slow = slow_matrix[1:, :]

# Compute percentage change; multiply by 100 to get percent.
pct_fast = np.divide(
    next_fast - prev_fast,
    prev_fast,
    out=np.full_like(next_fast, np.nan),
    where=(prev_fast != 0)
) * 100

pct_slow = np.divide(
    next_slow - prev_slow,
    prev_slow,
    out=np.full_like(next_slow, np.nan),
    where=(prev_slow != 0)
) * 100

# 5) Build DataFrames for percentage change:
years = list(range(start_year, start_year + (n_years - 1)))
# → [2007, 2008, …, 2099]

month_cols = [f"month{m}" for m in range(1, 13)]
df_fast_pct = pd.DataFrame(pct_fast, columns=month_cols, index=years)
df_fast_pct.insert(0, "year", df_fast_pct.index)

df_slow_pct = pd.DataFrame(pct_slow, columns=month_cols, index=years)
df_slow_pct.insert(0, "year", df_slow_pct.index)

# 6) Save each to CSV
output_fast_csv = OUTPUT_DIR / "cfast_yearly_pct_diff.csv"
output_slow_csv = OUTPUT_DIR / "cslow_yearly_pct_diff.csv"
df_fast_pct.to_csv(output_fast_csv, index=False)
df_slow_pct.to_csv(output_slow_csv, index=False)

print(f"Saved cfast percentage‐change to: {output_fast_csv}")
print(f"Saved cslow percentage‐change to: {output_slow_csv}")
