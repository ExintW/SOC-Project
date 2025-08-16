#!/usr/bin/env python3
import sys, os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from globals import PROCESSED_DIR, OUTPUT_DIR

OUT_SUBDIR = OUTPUT_DIR / 'Contribution'
OUT_SUBDIR.mkdir(exist_ok=True, parents=True)

# CSV paths
LAI_CSV = OUTPUT_DIR / 'annual_mean_lai_by_scenario.csv'
TP_CSV  = OUTPUT_DIR / 'tp_1950-2100_mean_tp.csv'
DAM_CSV = OUTPUT_DIR / 'dam_storage_by_year.csv'
SOC_CSV = OUTPUT_DIR / 'annual_mean_soc_by_scenario.csv'
STL_CSV = OUTPUT_DIR / 'stl_1950-2100_mean_temperature.csv'

years = np.arange(1950, 2024)

# =============================================================================
# Utilities: Sequential MK break detection
# =============================================================================
def sequential_mk(x):
    n = len(x)
    UF = np.zeros(n)
    for k in range(1, n):
        s = np.sum(np.sign(x[k] - x[:k]))
        E = k * (k - 1) / 4
        V = k * (k - 1) * (2 * k + 5) / 72
        UF[k] = (s - E) / np.sqrt(V) if V > 0 else 0.0
    y = x[::-1]
    UB_rev = np.zeros(n)
    for k in range(1, n):
        s = np.sum(np.sign(y[k] - y[:k]))
        E = k * (k - 1) / 4
        V = k * (k - 1) * (2 * k + 5) / 72
        UB_rev[k] = (s - E) / np.sqrt(V) if V > 0 else 0.0
    return UF, UB_rev[::-1]

def detect_one_mk_break(x):
    UF, UB = sequential_mk(x)
    diff = np.abs(UF - UB)
    return np.argmin(diff[1:-1]) + 1  # avoid endpoints

# =============================================================================
# Load time series
# =============================================================================
def load_df():
    # SOC
    df = pd.read_csv(SOC_CSV)
    soc = pd.Series(index=years, dtype=float)
    for y in years:
        m = (df.year == y) & df.scenario.isin(['Past', 'Present'])
        soc[y] = df.loc[m, 'mean'].squeeze() if not df.loc[m].empty else np.nan

    # LAI
    df = pd.read_csv(LAI_CSV); lai = pd.Series(index=years, dtype=float)
    for y in years:
        pat = 'Historical' if y <= 2000 else ('Present' if y <= 2014 else 'ssp245')
        m = (df.year == y) & df.scenario.str.contains(pat, case=False)
        lai[y] = df.loc[m, 'annual_mean_lai'].squeeze() if not df.loc[m].empty else np.nan

    # TP
    df = pd.read_csv(TP_CSV); tp = pd.Series(index=years, dtype=float)
    for y in years:
        scen = 'Past' if y <= 2006 else 'Present'
        m = (df.year == y) & (df.scenario == scen)
        tp[y] = df.loc[m, 'tp'].squeeze() if not df.loc[m].empty else np.nan

    # STL
    df = pd.read_csv(STL_CSV); stl = pd.Series(index=years, dtype=float)
    for y in years:
        scen = 'Past' if y <= 2006 else ('Present' if y <= 2014 else 'ssp245')
        m = (df.year == y) & (df.scenario == scen)
        stl[y] = df.loc[m, 'stl'].squeeze() if not df.loc[m].empty else np.nan

    # DAM → 2010+ use constant mean of 1950–2009
    df_dam = pd.read_csv(DAM_CSV).set_index('year')['total_storage_10k_m3']
    annual_dam = df_dam.reindex(years)
    mean_add = annual_dam.loc[1950:2009].mean()
    annual_dam.loc[2010:2100] = mean_add
    dam = annual_dam.fillna(0)

    return pd.DataFrame({
        'soc': soc, 'lai': lai, 'tp': tp, 'dam': dam, 'stl': stl
    }).dropna()

# =============================================================================
# Build 2-way interactions
# =============================================================================
def make_interactions(df, drivers):
    X = df[drivers].copy()
    for i in range(len(drivers)):
        for j in range(i + 1, len(drivers)):
            a, b = drivers[i], drivers[j]
            X[f'{a}x{b}'] = df[a] * df[b]
    return X

# =============================================================================
# VIP calculation
# =============================================================================
def pls_vip_contrib(X, y, n_components=2):
    # too few samples → return NaNs to avoid crash
    if X.shape[0] < 3 or np.allclose(np.std(y), 0):
        return np.full(X.shape[1], np.nan)

    scalerX = StandardScaler()
    Xc = scalerX.fit_transform(X)

    # center y only (keep units scale-neutral in VIP)
    yc = (y - np.mean(y))

    A = min(n_components, Xc.shape[0] - 1, Xc.shape[1])
    if A < 1:
        return np.full(X.shape[1], np.nan)

    pls = PLSRegression(n_components=A).fit(Xc, yc)

    W = pls.x_weights_             # (p, A)
    U = pls.y_scores_              # (n, A)
    SSY = np.sum(U ** 2, axis=0)   # variance explained per component
    p = W.shape[0]; total_SSY = SSY.sum()

    if total_SSY <= 0:
        return np.full(p, np.nan)

    VIP = np.zeros(p)
    for i in range(p):
        s = 0.0
        for a in range(A):
            wa = W[:, a]; wia = W[i, a]
            denom = np.dot(wa, wa)
            s += SSY[a] * (wia ** 2 / denom if denom > 0 else 0.0)
        VIP[i] = np.sqrt(p * s / total_SSY)

    sVIP = np.nansum(VIP)
    return (VIP / sVIP * 100) if sVIP > 0 else np.full(p, np.nan)

# =============================================================================
# Main: TWO change points (no cp3)
#   - cp1 from 1950..1995 window
#   - cp2 from 1996..2005 window
#   - For each cp, compute PLS VIP before/after within its window
# =============================================================================
def main():
    df = load_df()
    yrs = df.index.values
    soc = df['soc'].values

    # Resolve window boundaries safely
    if not np.any(yrs <= 1995) or not np.any((yrs >= 1996) & (yrs <= 2005)):
        raise ValueError("Not enough data to locate two windows (<=1995 and 1996–2005).")

    i1 = np.where(yrs <= 1995)[0][-1]          # end idx of first window
    i2 = np.where(yrs <= 2005)[0][-1]          # end idx of second window

    # Detect two change points
    cp1 = detect_one_mk_break(soc[:i1 + 1])                     # within [0..i1]
    cp2 = detect_one_mk_break(soc[i1 + 1:i2 + 1]) + (i1 + 1)    # within [i1+1..i2]

    cp_idxs  = [cp1, cp2]
    cp_years = yrs[cp_idxs]

    print("Detected Change Points (2 only):")
    for k, y in enumerate(cp_years, start=1):
        print(f"CP{k} at year {y}")

    drivers = ['lai', 'tp', 'dam', 'stl']
    rows = []

    for idx, cp in enumerate(cp_idxs):
        # segment bounds for this cp
        start = 0 if idx == 0 else (cp_idxs[idx - 1] + 1)
        end   = cp

        # after-segment ends at next cp start, else series end
        seg_end = (cp_idxs[idx + 1] + 1) if idx < len(cp_idxs) - 1 else len(yrs)

        seg_df_b = df.iloc[start: end + 1]
        seg_df_a = df.iloc[end + 1: seg_end]

        # Build features (same columns for b/a)
        feat_names = list(make_interactions(seg_df_b, drivers).columns)

        Xb = make_interactions(seg_df_b, drivers).values
        yb = seg_df_b['soc'].values
        Xa = make_interactions(seg_df_a, drivers).values
        ya = seg_df_a['soc'].values

        cb = pls_vip_contrib(Xb, yb)
        ca = pls_vip_contrib(Xa, ya)

        for name, vb, va in zip(feat_names, cb, ca):
            rows.append({
                'break': idx + 1,
                'year': int(cp_years[idx]),
                'period': 'before',
                'feature': name,
                'contrib_%': vb
            })
            rows.append({
                'break': idx + 1,
                'year': int(cp_years[idx]),
                'period': 'after',
                'feature': name,
                'contrib_%': va
            })

    out = pd.DataFrame(rows)
    out_path = OUT_SUBDIR / 'pls_interaction_vip_1950_2024.csv'
    out.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

if __name__ == '__main__':
    main()
