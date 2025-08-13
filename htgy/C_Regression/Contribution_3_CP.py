#!/usr/bin/env python3
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from globals import PROCESSED_DIR, OUTPUT_DIR

OUT_SUBDIR = OUTPUT_DIR / 'Contribution'
OUT_SUBDIR.mkdir(exist_ok=True, parents=True)

LAI_CSV = OUTPUT_DIR / 'annual_mean_lai_by_scenario.csv'
TP_CSV  = OUTPUT_DIR / 'tp_1950-2100_mean_tp.csv'
DAM_CSV = OUTPUT_DIR / 'dam_storage_by_year.csv'
SOC_CSV = OUTPUT_DIR / 'annual_mean_soc_by_scenario.csv'
STL_CSV = OUTPUT_DIR / 'stl_1950-2100_mean_temperature.csv'

# use the same years array as before (1950–2023)
years = np.arange(1950, 2024)

# =============================================================================
# MANN–KENDALL SEQUENTIAL TEST (single‐break)
# =============================================================================
def sequential_mk(x):
    x = np.asarray(x, float)
    n = len(x)
    UF = np.zeros(n)
    for k in range(1, n):
        s = sum(np.sign(x[k] - x[i]) for i in range(k))
        E = k * (k - 1) / 4
        Var = k * (k - 1) * (2*k + 5) / 72
        UF[k] = (s - E) / np.sqrt(Var) if Var > 0 else 0

    y = x[::-1]
    UB_rev = np.zeros(n)
    for k in range(1, n):
        s = sum(np.sign(y[k] - y[i]) for i in range(k))
        E = k * (k - 1) / 4
        Var = k * (k - 1) * (2*k + 5) / 72
        UB_rev[k] = (s - E) / np.sqrt(Var) if Var > 0 else 0

    UB = UB_rev[::-1]
    return UF, UB

def detect_one_mk_break(x):
    """Return the single change‐point index (0‐based) in x via |UF−UB| min."""
    UF, UB = sequential_mk(x)
    diff = np.abs(UF - UB)
    # ignore first & last
    idx = np.argmin(diff[1:-1]) + 1
    return idx

# =============================================================================
# LOAD & BUILD ANNUAL SERIES
# =============================================================================
def load_series():
    # SOC
    df = pd.read_csv(SOC_CSV)
    soc = pd.Series(index=years, dtype=float)
    for y in years:
        m = (df.year == y) & df.scenario.isin(['Past', 'Present'])
        soc[y] = df.loc[m, 'mean'].squeeze() if not df.loc[m].empty else np.nan

    # LAI
    df = pd.read_csv(LAI_CSV)
    lai = pd.Series(index=years, dtype=float)
    for y in years:
        pat = 'Historical' if y <= 2000 else ('Present' if y <= 2014 else 'ssp245')
        m = (df.year == y) & df.scenario.str.contains(pat)
        lai[y] = df.loc[m, 'annual_mean_lai'].squeeze() if not df.loc[m].empty else np.nan

    # TP
    df = pd.read_csv(TP_CSV)
    tp = pd.Series(index=years, dtype=float)
    for y in years:
        scen = 'Past' if y <= 2006 else 'Present'
        m = (df.year == y) & (df.scenario == scen)
        tp[y] = df.loc[m, 'tp'].squeeze() if not df.loc[m].empty else np.nan

    # STL
    df = pd.read_csv(STL_CSV)
    stl = pd.Series(index=years, dtype=float)
    for y in years:
        scen = 'Past' if y <= 2006 else ('Present' if y <= 2014 else 'ssp245')
        m = (df.year == y) & (df.scenario == scen)
        stl[y] = df.loc[m, 'stl'].squeeze() if not df.loc[m].empty else np.nan

    # DAM
    df_dam = pd.read_csv(DAM_CSV).set_index('year')['total_storage_10k_m3']
    dam = df_dam.reindex(years)
    ref = dam.loc[2004:2009].mean()
    dam.loc[2010:] = ref
    dam = dam.fillna(0)

    return pd.DataFrame({'soc': soc, 'lai': lai, 'tp': tp, 'dam': dam, 'stl': stl}).dropna()

# =============================================================================
# MAIN
# =============================================================================
def main():
    df = load_series()
    yrs = df.index.values
    soc_vals = df['soc'].values

    # define slice boundaries by position
    i1_end = np.where(yrs <= 1995)[0][-1]
    i2_start, i2_end = i1_end + 1, np.where(yrs <= 2005)[0][-1]
    i3_start = i2_end + 1
    # i3_end is implicitly len(yrs)-1

    # detect one MK break per period
    cp1_rel = detect_one_mk_break(soc_vals[0:i1_end+1])
    cp2_rel = detect_one_mk_break(soc_vals[i2_start:i2_end+1])
    cp3_rel = detect_one_mk_break(soc_vals[i3_start:])

    # convert to full‐series indices
    cp_idxs = np.array([
        cp1_rel,
        i2_start + cp2_rel,
        i3_start + cp3_rel
    ])
    cp_years = yrs[cp_idxs]
    print("Change‐points:", list(cp_years))

    # build cumulative series
    cum = df.cumsum()
    bounds = [0, cp_idxs[0], cp_idxs[1], cp_idxs[2], len(yrs)]

    # fit linear trends on each segment
    regs = {}
    for var in ['soc', 'lai', 'tp', 'dam', 'stl']:
        segs = []
        for j in range(len(bounds)-1):
            a, b = bounds[j], bounds[j+1]
            xseg, yseg = yrs[a:b], cum[var].values[a:b]
            m, intercept, *_ = linregress(xseg, yseg)
            segs.append((m, intercept))
        regs[var] = segs

    # compute slope‐change rates & contributions
    rows = []
    for j in range(3):
        # SOC slope change at break j
        m0_soc = regs['soc'][j][0]
        m1_soc = regs['soc'][j+1][0]
        delta_soc = m1_soc - m0_soc
        rate_soc = delta_soc / m0_soc * 100

        for var in ['soc', 'lai', 'tp', 'dam', 'stl']:
            m0, m1 = regs[var][j][0], regs[var][j+1][0]
            rate_var = (m1 - m0) / m0 * 100
            contrib = 100.0 if var == 'soc' else (rate_var / rate_soc) * 100
            rows.append({
                'break_number': j+1,
                'change_year': int(cp_years[j]),
                'variable': var,
                'slope_before': m0,
                'slope_after': m1,
                'slope_change_rate_%': rate_var,
                'contribution_%': contrib
            })

    pd.DataFrame(rows).to_csv(OUT_SUBDIR / 'contribution_rate_3periods.csv', index=False)
    print("Saved → contribution_rate_3periods.csv")

    # optional: plot results
    for var in ['soc', 'lai', 'tp', 'dam', 'stl']:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(yrs, cum[var], 'o', label=f'Cumulative {var.upper()}')
        for j in range(len(bounds)-1):
            a, b = bounds[j], bounds[j+1]
            m, c = regs[var][j]
            ax.plot(yrs[a:b], m*yrs[a:b] + c, '-', label=f'Segment {j+1}')
        for y in cp_years:
            ax.axvline(y, color='k', linestyle='--')
        ax.set_title(f'{var.upper()} — 3 MK Change‐Points by Period')
        ax.set_xlabel('Year'); ax.set_ylabel(f'Cumulative {var.upper()}')
        ax.legend()
        fig.savefig(OUT_SUBDIR / f'{var}_3periods.png', dpi=300)
        plt.close(fig)

    print("Plots saved in Contribution folder.")

if __name__ == '__main__':
    main()
