#!/usr/bin/env python3
import sys, os
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# =============================================================================
# (1) CONFIGURATION & PATHS
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
# (2) Utilities: Sequential MK break detection
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
# (3) Load time series
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

    df_all = pd.DataFrame({'soc': soc, 'lai': lai, 'tp': tp, 'dam': dam, 'stl': stl}).dropna()
    df_all.index.name = 'year'
    return df_all

# =============================================================================
# (4) Feature builders
# =============================================================================
def singles(df, drivers):
    return df[drivers].copy()

def singles_plus_exact_k(df, drivers, k):
    X = singles(df, drivers)  # always include singles
    for combo in combinations(drivers, k):
        name = "x".join(combo)
        prod = np.prod([df[c] for c in combo], axis=0)
        X[name] = prod
    return X

# sets as requested
def features_set1(df, drivers):  # singles + only 2-way
    return singles_plus_exact_k(df, drivers, 2)

def features_set2(df, drivers):  # singles + only 3-way
    return singles_plus_exact_k(df, drivers, 3)

def features_set3(df, drivers):  # singles + only 4-way
    return singles_plus_exact_k(df, drivers, 4)

# =============================================================================
# (5) VIP calculation
# =============================================================================
def pls_vip_contrib(X, y, n_components=2):
    if X.shape[0] < 3 or np.allclose(np.std(y), 0) or X.shape[1] == 0:
        return np.full(X.shape[1], np.nan)

    Xc = StandardScaler().fit_transform(X)
    yc = (y - np.mean(y))

    A = min(n_components, Xc.shape[0] - 1, Xc.shape[1])
    if A < 1:
        return np.full(X.shape[1], np.nan)

    pls = PLSRegression(n_components=A).fit(Xc, yc)
    W = pls.x_weights_             # (p, A)
    U = pls.y_scores_              # (n, A)
    SSY = np.sum(U ** 2, axis=0)
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
# (6) Main: CPs → 3 segments → compute 3 sets → wide layout (3 columns)
# =============================================================================
def main():
    df = load_df()
    yrs = df.index.values
    soc = df['soc'].values

    # detect CPs in fixed windows
    if not np.any(yrs <= 1995) or not np.any((yrs >= 1996) & (yrs <= 2005)):
        raise ValueError("Not enough data to locate two windows (<=1995 and 1996–2005).")
    i1 = np.where(yrs <= 1995)[0][-1]
    i2 = np.where(yrs <= 2005)[0][-1]

    cp1 = detect_one_mk_break(soc[:i1 + 1])
    cp2 = detect_one_mk_break(soc[i1 + 1:i2 + 1]) + (i1 + 1)
    print(f"Detected CP1={int(yrs[cp1])}, CP2={int(yrs[cp2])}")

    # segments (no before/after duplication)
    segments = [
        ("S1", 0,        cp1),             # start .. CP1
        ("S2", cp1 + 1,  cp2),             # CP1+1 .. CP2
        ("S3", cp2 + 1,  len(yrs) - 1),    # CP2+1 .. end
    ]

    drivers = ['lai', 'tp', 'dam', 'stl']
    builders = {
        'set1_2way': features_set1,  # singles + 2-way
        'set2_3way': features_set2,  # singles + 3-way
        'set3_4way': features_set3,  # singles + 4-way
    }

    # Collect tidy then pivot to wide
    tidy_rows = []
    for set_name, builder in builders.items():
        for seg_label, a, b in segments:
            if a > b:
                continue
            seg_df = df.iloc[a:b+1]
            X_df = builder(seg_df, drivers)
            feat_names = list(X_df.columns)
            contrib = pls_vip_contrib(X_df.values, seg_df['soc'].values)
            seg_years = f"{int(yrs[a])}-{int(yrs[b])}"

            for name, v in zip(feat_names, contrib):
                tidy_rows.append({
                    'set': set_name,
                    'segment': seg_label,
                    'segment_years': seg_years,
                    'feature': name,
                    'contrib_%': v
                })

    out_tidy = pd.DataFrame(tidy_rows)

    # Wide layout: one row per segment×feature, with separate columns per set
    wide = (
        out_tidy
        .pivot_table(
            index=['segment', 'segment_years', 'feature'],
            columns='set',
            values='contrib_%',
            aggfunc='first'
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # column order
    cols = ['segment', 'segment_years', 'feature', 'set1_2way', 'set2_3way', 'set3_4way']
    wide = wide.reindex(columns=[c for c in cols if c in wide.columns])

    out_path = OUT_SUBDIR / 'pls_interaction_vip_1950_2024.csv'
    wide.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

if __name__ == '__main__':
    main()
