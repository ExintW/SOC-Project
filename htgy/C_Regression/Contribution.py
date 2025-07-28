import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, chi2

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR

# dedicate a subfolder for outputs
OUT_SUBDIR = OUTPUT_DIR / "Contribution"
OUT_SUBDIR.mkdir(exist_ok=True, parents=True)

# input annual CSVs
LAI_CSV = OUTPUT_DIR / "annual_mean_lai_by_scenario.csv"
TP_CSV  = OUTPUT_DIR / "tp_1950-2100_mean_tp.csv"
DAM_CSV = OUTPUT_DIR / "dam_storage_by_year.csv"
SOC_CSV = OUTPUT_DIR / "annual_mean_soc_by_scenario.csv"
STL_CSV = OUTPUT_DIR / "stl_1950-2100_mean_temperature.csv"   # newly added

# years of interest
years = np.arange(1950, 2024)

# =============================================================================
# SNHT (Standard Normal Homogeneity Test) for a single change point
# =============================================================================
def snht_change_point(x):
    """
    Returns (cp_index, p_value) for a single change point in x using SNHT.
    cp_index is 0-based break index; p_value is approximate significance.
    """
    x = np.asarray(x, float)
    n = len(x)
    mu = x.mean()
    s2 = x.var(ddof=1)
    cumsum = np.cumsum(x)
    T = np.zeros(n-1, float)
    for k in range(1, n):
        mean1 = cumsum[k-1] / k
        mean2 = (cumsum[-1] - cumsum[k-1]) / (n-k)
        T[k-1] = (k*(mean1-mu)**2 + (n-k)*(mean2-mu)**2) / s2
    idx = int(np.argmax(T))
    Tk = T[idx]
    p_value = 1 - chi2.cdf(Tk, df=1)
    return idx+1, p_value  # break at index+1

# =============================================================================
# Load and build annual series for SOC, LAI, TP, Dam, STL
# =============================================================================

# 1) SOC
df_soc = pd.read_csv(SOC_CSV)
soc = pd.Series(index=years, dtype=float)
for y in years:
    mask = (df_soc.year == y) & (df_soc.scenario.isin(['Past', 'Present']))
    vals = df_soc.loc[mask, 'mean']
    soc[y] = vals.values[0] if not vals.empty else np.nan

# 2) LAI
df_lai = pd.read_csv(LAI_CSV)
lai = pd.Series(index=years, dtype=float)
for y in years:
    if y <= 2000:
        pattern = 'Historical'
    elif y <= 2014:
        pattern = 'Present'
    else:
        pattern = 'ssp245'
    mask = (df_lai.year == y) & df_lai.scenario.str.contains(pattern)
    vals = df_lai.loc[mask, 'annual_mean_lai']
    lai[y] = vals.values[0] if not vals.empty else np.nan

# 3) Precipitation (TP)
df_tp = pd.read_csv(TP_CSV)
tp = pd.Series(index=years, dtype=float)
for y in years:
    scen = 'Past' if y <= 2006 else 'Present'
    vals = df_tp.loc[(df_tp.year == y) & (df_tp.scenario == scen), 'tp']
    tp[y] = vals.values[0] if not vals.empty else np.nan

# 4) STL (Soil Temperature)
df_stl = pd.read_csv(STL_CSV)
stl = pd.Series(index=years, dtype=float)
for y in years:
    if y <= 2006:
        scen = 'Past'
    elif y <= 2014:
        scen = 'Present'
    else:
        scen = 'ssp245'
    vals = df_stl.loc[
        (df_stl.year == y) & (df_stl.scenario == scen),
        'stl'
    ]
    stl[y] = vals.iloc[0] if not vals.empty else np.nan

# 5) Dam storage
df_dam = pd.read_csv(DAM_CSV).set_index('year')['total_storage_10k_m3']
annual_dam = df_dam.reindex(years)
mean_add = annual_dam.loc[1980:2009].mean()
annual_dam.loc[2010:2024] = mean_add
annual_dam = annual_dam.fillna(0)
dam = annual_dam

# Combine into DataFrame keeping only years where all drivers exist
df = pd.DataFrame({
    'soc': soc,
    'lai': lai,
    'tp': tp,
    'dam': dam,
    'stl': stl
})
df = df.dropna(subset=['soc', 'lai', 'tp', 'dam', 'stl'])
yrs = df.index.values

# =============================================================================
# Detect change point in SOC via SNHT
# =============================================================================
cp_idx, p_val = snht_change_point(df['soc'].values)
cp_year = yrs[cp_idx]
print(f"SNHT change point at year {cp_year} (index {cp_idx}), p = {p_val:.3f}")

# =============================================================================
# Build cumulative series & regress before/after cp
# =============================================================================
cum = df.cumsum()

def fit_reg(series):
    vals = series.values
    x1 = yrs[yrs <= cp_year]; y1 = vals[yrs <= cp_year]
    x2 = yrs[yrs > cp_year];  y2 = vals[yrs > cp_year]
    m1, b1, *_ = linregress(x1, y1)
    m2, b2, *_ = linregress(x2, y2)
    return (m1, b1), (m2, b2)

regs = {v: fit_reg(cum[v]) for v in ['soc', 'lai', 'tp', 'dam', 'stl']}

# =============================================================================
# Compute slope-change rates & contributions
# =============================================================================
R = {
    v: (regs[v][1][0] - regs[v][0][0]) / regs[v][0][0] * 100
    for v in regs
}
R_soc = R['soc']
C = {
    v: (R[v] / R_soc) * 100
    for v in ['lai', 'tp', 'dam', 'stl']
}

# =============================================================================
# Save summary CSV
# =============================================================================
rows = []
for v, ((m1, _), (m2, _)) in regs.items():
    rows.append({
        'variable': v,
        'change_year': int(cp_year),
        'slope_before': m1,
        'slope_after': m2,
        'slope_change_rate_%': R[v],
        'contribution_%': (100 if v == 'soc' else C[v])
    })
pd.DataFrame(rows).to_csv(OUT_SUBDIR / "contribution_rate_summary.csv", index=False)

# =============================================================================
# Plot regressions with change point
# =============================================================================
for v in ['soc', 'lai', 'tp', 'dam', 'stl']:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(yrs, cum[v], 'o', label=f'Cumulative {v.upper()}')
    (m1, b1), (m2, b2) = regs[v]
    ax.plot(yrs[yrs <= cp_year], m1 * yrs[yrs <= cp_year] + b1,
            '-', label=f'y={m1:.3f}x+{b1:.1f}')
    ax.plot(yrs[yrs > cp_year], m2 * yrs[yrs > cp_year] + b2,
            '-', label=f'y={m2:.3f}x+{b2:.1f}')
    ax.axvline(cp_year, color='k', linestyle='--',
               label=f'Change @ {cp_year}')
    ax.set_title(f'Regression for {v.upper()}')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Cumulative {v.upper()}')
    ax.legend()
    fig.savefig(OUT_SUBDIR / f"{v}_regression_with_cp.png", dpi=300)
    plt.close(fig)

print("Analysis complete. Summary CSV and plots saved in Contribution folder.")
