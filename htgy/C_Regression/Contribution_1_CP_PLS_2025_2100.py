#!/usr/bin/env python3
import sys, os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 1) CONFIG
# ─────────────────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR

OUT_SUBDIR = OUTPUT_DIR / "Contribution"
OUT_SUBDIR.mkdir(exist_ok=True, parents=True)

LAI_CSV = OUTPUT_DIR / "annual_mean_lai_by_scenario.csv"
TP_CSV  = OUTPUT_DIR / "tp_1950-2100_mean_tp.csv"
DAM_CSV = OUTPUT_DIR / "dam_storage_by_year.csv"
SOC_CSV = OUTPUT_DIR / "annual_mean_soc_by_scenario.csv"
STL_CSV = OUTPUT_DIR / "stl_1950-2100_mean_temperature.csv"

years = np.arange(2025, 2101)  # focus only on 2025–2100
SCEN  = "ssp245"               # scenario used for 2025–2100

# ─────────────────────────────────────────────────────────────────────────────
# 2) MK change point (single CP)
# ─────────────────────────────────────────────────────────────────────────────
def sequential_mk(x):
    """
    Sequential Mann–Kendall test returning 1 CP.
    Returns: idx (0-based), UF, UB
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return 0, np.full(n, np.nan), np.full(n, np.nan)

    UF = np.full(n, np.nan)
    # forward
    for k in range(1, n):
        s = 0
        for i in range(k):
            s += 1 if x[k] > x[i] else (-1 if x[k] < x[i] else 0)
        var_s = k*(k-1)*(2*k+5)/18.0
        UF[k] = s/np.sqrt(var_s) if var_s > 0 else np.nan

    # backward on reversed
    x_rev = x[::-1]
    UB_rev = np.full(n, np.nan)
    for k in range(1, n):
        s = 0
        for i in range(k):
            s += 1 if x_rev[k] > x_rev[i] else (-1 if x_rev[k] < x_rev[i] else 0)
        var_s = k*(k-1)*(2*k+5)/18.0
        UB_rev[k] = s/np.sqrt(var_s) if var_s > 0 else np.nan

    UB = UB_rev[::-1]
    idx = int(np.nanargmin(np.abs(UF - UB)))
    # keep CP interior so both sides have data
    idx = max(1, min(n-2, idx))
    return idx, UF, UB

# ─────────────────────────────────────────────────────────────────────────────
# 3) helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_interactions(df, drivers):
    X = df[drivers].copy()
    cols = list(drivers)
    for i in range(len(drivers)):
        for j in range(i+1, len(drivers)):
            a, b = drivers[i], drivers[j]
            name = f"{a}x{b}"
            X[name] = df[a] * df[b]
            cols.append(name)
    return X[cols]

def pls_vip_contrib(X, y, n_components=2):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    A = min(n_components, p, max(1, n-1))
    if n < 3 or p < 1:
        return np.full(p, np.nan)

    scalerX = StandardScaler()
    Xc = scalerX.fit_transform(X)
    scalerY = StandardScaler(with_std=False)
    yc = scalerY.fit_transform(y.reshape(-1,1)).ravel()

    pls = PLSRegression(n_components=A)
    pls.fit(Xc, yc)

    W = pls.x_weights_          # (p, A)
    U = pls.y_scores_           # (n, A)
    SSY = np.sum(U**2, axis=0)  # (A,)
    total_SSY = SSY.sum()

    VIP = np.zeros(p)
    for i in range(p):
        s = 0.0
        for a in range(A):
            wa = W[:, a]
            wia = W[i, a]
            denom = np.dot(wa, wa)
            if denom > 0:
                s += SSY[a] * (wia**2 / denom)
        VIP[i] = np.sqrt(p * s / total_SSY) if total_SSY > 0 else np.nan

    if np.all(~np.isnan(VIP)) and VIP.sum() != 0:
        VIP = VIP / VIP.sum() * 100.0
    return VIP

# ─────────────────────────────────────────────────────────────────────────────
# 4) load series for 2025–2100 (mimics your example’s extraction)
# ─────────────────────────────────────────────────────────────────────────────
def load_series_2025_2100():
    # SOC (exact match to SCEN)
    df_soc = pd.read_csv(SOC_CSV)
    df_soc['scenario'] = df_soc['scenario'].astype(str).str.strip()
    soc = pd.Series(index=years, dtype=float)
    for y in years:
        vals = df_soc.loc[(df_soc.year == y) & (df_soc.scenario == SCEN), 'mean']
        soc[y] = vals.iloc[0] if not vals.empty else np.nan

    # LAI (files often say "Historical/Present/ssp245 (...)" → use contains)
    df_lai = pd.read_csv(LAI_CSV)
    df_lai['scenario'] = df_lai['scenario'].astype(str).str.strip()
    lai_col = 'annual_mean_lai' if 'annual_mean_lai' in df_lai.columns else (
              'lai' if 'lai' in df_lai.columns else 'mean')
    lai = pd.Series(index=years, dtype=float)
    for y in years:
        mask = (df_lai.year == y) & (df_lai.scenario.str.contains(SCEN, case=False))
        vals = df_lai.loc[mask, lai_col]
        lai[y] = vals.iloc[0] if not vals.empty else np.nan

    # TP (use SCEN for 2025+ like your example)
    df_tp = pd.read_csv(TP_CSV)
    df_tp['scenario'] = df_tp['scenario'].astype(str).str.strip()
    tp = pd.Series(index=years, dtype=float)
    for y in years:
        vals = df_tp.loc[(df_tp.year == y) & (df_tp.scenario == SCEN), 'tp']
        tp[y] = vals.iloc[0] if not vals.empty else np.nan

    # STL (use SCEN for 2025+ like your example)
    df_stl = pd.read_csv(STL_CSV)
    df_stl['scenario'] = df_stl['scenario'].astype(str).str.strip()
    stl = pd.Series(index=years, dtype=float)
    for y in years:
        vals = df_stl.loc[(df_stl.year == y) & (df_stl.scenario == SCEN), 'stl']
        stl[y] = vals.iloc[0] if not vals.empty else np.nan

    # DAM (constant mean method; extend to 2100)
    df_dam = pd.read_csv(DAM_CSV).set_index('year')['total_storage_10k_m3']
    annual_dam = df_dam.reindex(np.arange(1950, 2101))
    mean_add = annual_dam.loc[1950:2009].mean()
    annual_dam.loc[2010:2100] = mean_add
    dam = annual_dam.reindex(years).fillna(0)

    # show availability before we drop rows
    print("Data availability 2025–2100 (non-NA counts):")
    for name, s in [('SOC', soc), ('LAI', lai), ('TP', tp), ('DAM', dam), ('STL', stl)]:
        print(f"  {name:<3}: {s.count():>3} / {len(years)}")

    df = pd.DataFrame({'soc': soc, 'lai': lai, 'tp': tp, 'dam': dam, 'stl': stl})
    df = df.dropna(subset=['soc','lai','tp','dam','stl'])
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 5) main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = load_series_2025_2100()
    if df.empty or len(df) < 6:
        print("Not enough complete rows after dropna() to run CP + PLS.")
        return

    yrs = df.index.values
    soc = df['soc'].values

    # single change point (interior)
    cp_idx, UF, UB = sequential_mk(soc)
    cp_year = int(yrs[cp_idx])
    print(f"Sequential MK change point at year {cp_year} (index {cp_idx} within 2025–2100 subset)")

    drivers = ['lai', 'tp', 'dam', 'stl']

    # split before/after
    seg_b = df.iloc[:cp_idx+1].copy()
    seg_a = df.iloc[cp_idx+1:].copy()

    # build interactions
    Xb = make_interactions(seg_b, drivers)
    Xa = make_interactions(seg_a, drivers)
    yb = seg_b['soc'].values
    ya = seg_a['soc'].values

    # compute VIP contributions
    cb = pls_vip_contrib(Xb.values, yb)
    ca = pls_vip_contrib(Xa.values, ya)
    feat_names = list(Xb.columns)

    # pack results
    rows = []
    for name, vb, va in zip(feat_names, cb, ca):
        rows.append({'break': 1, 'year': cp_year, 'period': 'before', 'feature': name, 'contrib_%': vb})
        rows.append({'break': 1, 'year': cp_year, 'period': 'after',  'feature': name, 'contrib_%': va})

    out = pd.DataFrame(rows)
    out_path = OUT_SUBDIR / "pls_interaction_vip_2025_2100.csv"
    out.to_csv(out_path, index=False)
    print("Saved →", out_path)

if __name__ == "__main__":
    main()
