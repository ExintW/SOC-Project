import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =============================================================================
# 1) CONFIGURATION & PATHS
# =============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR

OUT_SUBDIR = OUTPUT_DIR / "Contribution"
OUT_SUBDIR.mkdir(exist_ok=True, parents=True)

LAI_CSV = OUTPUT_DIR / "annual_mean_lai_by_scenario.csv"
TP_CSV  = OUTPUT_DIR / "tp_1950-2100_mean_tp.csv"
DAM_CSV = OUTPUT_DIR / "dam_storage_by_year.csv"
SOC_CSV = OUTPUT_DIR / "annual_mean_soc_by_scenario.csv"
STL_CSV = OUTPUT_DIR / "stl_1950-2100_mean_temperature.csv"

# now cover 1950 through 2100 inclusive
years = np.arange(1950, 2101)

# =============================================================================
# 2) SEQUENTIAL MANN–KENDALL CHANGE‑POINT DETECTION
# =============================================================================
def sequential_mk(x):
    """
    Sequential Mann–Kendall test to find a single change point.
    Returns: idx (0‑based), UF, UB
    """
    n = len(x)
    UF = np.full(n, np.nan)
    UB = np.full(n, np.nan)

    # forward
    S = np.zeros(n)
    for k in range(1, n):
        s = 0
        for i in range(k):
            if x[k] > x[i]:
                s += 1
            elif x[k] < x[i]:
                s -= 1
        S[k] = s
        var_s = k*(k-1)*(2*k+5)/18.0
        UF[k] = s/np.sqrt(var_s) if var_s>0 else np.nan

    # backward (on reversed series)
    x_rev = x[::-1]
    S_rev = np.zeros(n)
    UB_rev = np.full(n, np.nan)
    for k in range(1, n):
        s = 0
        for i in range(k):
            if x_rev[k] > x_rev[i]:
                s += 1
            elif x_rev[k] < x_rev[i]:
                s -= 1
        S_rev[k] = s
        var_s = k*(k-1)*(2*k+5)/18.0
        UB_rev[k] = s/np.sqrt(var_s) if var_s>0 else np.nan

    UB = UB_rev[::-1]
    idx = int(np.nanargmin(np.abs(UF - UB)))
    return idx, UF, UB

# =============================================================================
# 3) LOAD & BUILD ANNUAL SERIES (1950–2100)
# =============================================================================

# -- SOC ----------------------------------------------------------
df_soc = pd.read_csv(SOC_CSV)
soc = pd.Series(index=years, dtype=float)
for y in years:
    if y <= 2006:
        scen = 'Past'
    elif y <= 2024:
        scen = 'Present'
    else:
        scen = 'ssp245'
    vals = df_soc.loc[
        (df_soc.year == y) & (df_soc.scenario == scen),
        'mean'
    ]
    soc[y] = vals.iloc[0] if not vals.empty else np.nan

# ─── LOAD LAI ────────────────────────────────────────────────────────────────
df_lai = pd.read_csv(LAI_CSV)

lai = pd.Series(index=years, dtype=float)
for y in years:
    if   y <= 2000:
        scen = 'Historical'    # matches “Historical (1950 – 2000)”
    elif y <= 2014:
        scen = 'Present'       # matches “Present (2001 – 2014)”
    else:
        scen = 'ssp245'        # now matches “ssp245 (2015 – 2100)”
    mask = (
        (df_lai.year == y)
        & df_lai.scenario.str.contains(scen, case=False)
    )
    vals = df_lai.loc[mask, 'annual_mean_lai']
    lai[y] = vals.iloc[0] if not vals.empty else np.nan


# -- TP -----------------------------------------------------------
df_tp = pd.read_csv(TP_CSV)
tp = pd.Series(index=years, dtype=float)
for y in years:
    if y <= 2006:
        scen = 'Past'
    elif y <= 2024:
        scen = 'Present'
    else:
        scen = 'ssp245'
    vals = df_tp.loc[
        (df_tp.year == y) & (df_tp.scenario == scen),
        'tp'
    ]
    tp[y] = vals.iloc[0] if not vals.empty else np.nan

# -- STL ----------------------------------------------------------
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

# -- Dam storage (constant‑mean method) -------------------------
df_dam = pd.read_csv(DAM_CSV).set_index('year')['total_storage_10k_m3']
annual_dam = df_dam.reindex(years)
mean_add = annual_dam.loc[1950:2009].mean()
# apply constant mean from 2010 through 2100
annual_dam.loc[2010:2100] = mean_add
dam = annual_dam.fillna(0)


# DEBUG: print non‐NA count for each variable
print("Data availability (1950–2100):")
for name, series in [
    ('SOC', soc),
    ('LAI', lai),
    ('TP',  tp),
    ('DAM', dam),
    ('STL', stl)
]:
    print(f"  {name:<3}: {series.count():>3} non‐NA / {len(years)} total years")
print("——————————————————————————————————————————————————————————————\n")


# -- combine and drop missing ----------------------------------
df = pd.DataFrame({
    'soc': soc,
    'lai': lai,
    'tp':  tp,
    'dam': dam,
    'stl': stl
})
df = df.dropna(subset=['soc','lai','tp','dam','stl'])
yrs = df.index.values

# =============================================================================
# 4) CHANGE POINT VIA SEQUENTIAL MK
# =============================================================================
cp_idx, UF, UB = sequential_mk(df['soc'].values)
cp_year = yrs[cp_idx]
print(f"Sequential MK change point at year {cp_year} (index {cp_idx})")

# =============================================================================
# 5) CUMULATIVE SERIES & REGRESSION
# =============================================================================
cum = df.cumsum()
def fit_reg(series):
    vals = series.values
    x1 = yrs[yrs <= cp_year]; y1 = vals[yrs <= cp_year]
    x2 = yrs[yrs >  cp_year]; y2 = vals[yrs >  cp_year]
    m1, b1, *_ = linregress(x1, y1)
    m2, b2, *_ = linregress(x2, y2)
    return (m1, b1), (m2, b2)

regs = {v: fit_reg(cum[v]) for v in ['soc','lai','tp','dam','stl']}

# =============================================================================
# 6) SLOPE-CHANGE RATES & CONTRIBUTIONS
# =============================================================================
R     = {v: (regs[v][1][0] - regs[v][0][0]) / regs[v][0][0] * 100
         for v in regs}
R_soc = R['soc']
C     = {v: R[v]/R_soc*100 for v in ['lai','tp','dam','stl']}

# =============================================================================
# 7) SAVE SUMMARY CSV
# =============================================================================
rows = []
for v, ((m1,_),(m2,_)) in regs.items():
    rows.append({
        'variable': v,
        'change_year': int(cp_year),
        'slope_before': m1,
        'slope_after': m2,
        'slope_change_rate_%': R[v],
        'contribution_%': (100 if v=='soc' else C[v])
    })
pd.DataFrame(rows).to_csv(OUT_SUBDIR/"contribution_rate_summary.csv", index=False)

# =============================================================================
# 8) PLOT REGRESSIONS
# =============================================================================
for v in ['soc','lai','tp','dam','stl']:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(yrs, cum[v], 'o', label=f'Cumulative {v.upper()}')
    (m1,b1),(m2,b2) = regs[v]
    ax.plot(yrs[yrs<=cp_year], m1*yrs[yrs<=cp_year]+b1, '-',
            label=f'y={m1:.3f}x+{b1:.1f}')
    ax.plot(yrs[yrs> cp_year], m2*yrs[yrs> cp_year]+b2, '-',
            label=f'y={m2:.3f}x+{b2:.1f}')
    ax.axvline(cp_year, linestyle='--', label=f'Change @ {cp_year}')
    ax.set_title(f'Regression for {v.upper()}')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Cumulative {v.upper()}')
    ax.legend()
    fig.savefig(OUT_SUBDIR/f"{v}_regression_with_cp.png", dpi=300)
    plt.close(fig)

print("Done — summary CSV and plots in:", OUT_SUBDIR)
