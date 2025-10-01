#!/usr/bin/env python3
import sys, os
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 1) CONFIG
# ─────────────────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR

OUT_SUBDIR = OUTPUT_DIR / "Contribution_dam_rem"
OUT_SUBDIR.mkdir(exist_ok=True, parents=True)

LAI_CSV = OUTPUT_DIR / "annual_mean_lai_by_scenario.csv"
TP_CSV  = OUTPUT_DIR / "tp_1950-2100_mean_tp.csv"
SOC_CSV = OUTPUT_DIR / "annual_mean_soc_by_scenario.csv"
STL_CSV = OUTPUT_DIR / "stl_1950-2100_mean_temperature.csv"
# NEW: dam remained capacity (annual mean of monthly sums) by scenario
DAM_REM_CSV = OUTPUT_DIR / "annual_mean_dam_rem_cap_by_scenario.csv"

years = np.arange(2025, 2101)  # 2025–2100 inclusive
SCEN  = "ssp245"               # use ssp245 for all series in this window

# ─────────────────────────────────────────────────────────────────────────────
# 2) MK change point (single CP; interior)
# ─────────────────────────────────────────────────────────────────────────────
def sequential_mk_single_cp(x):
    """
    Sequential MK forward/backward; returns a single interior CP index.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return max(0, n // 2)

    UF = np.full(n, np.nan)
    for k in range(1, n):
        s = 0
        for i in range(k):
            s += 1 if x[k] > x[i] else (-1 if x[k] < x[i] else 0)
        var_s = k*(k-1)*(2*k+5)/18.0
        UF[k] = s/np.sqrt(var_s) if var_s > 0 else np.nan

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
    return max(1, min(n - 2, idx))  # force interior

# ─────────────────────────────────────────────────────────────────────────────
# 3) LOAD SERIES (2025–2100; use ssp245 for all, incl. DAM rem-cap)
# ─────────────────────────────────────────────────────────────────────────────
def load_series_2025_2100():
    # SOC
    df_soc = pd.read_csv(SOC_CSV)
    df_soc['scenario'] = df_soc['scenario'].astype(str).str.strip()
    soc = pd.Series(index=years, dtype=float)
    for y in years:
        vals = df_soc.loc[(df_soc.year == y) & (df_soc.scenario == SCEN), 'mean']
        soc[y] = vals.iloc[0] if not vals.empty else np.nan

    # LAI (column name can vary)
    df_lai = pd.read_csv(LAI_CSV)
    df_lai['scenario'] = df_lai['scenario'].astype(str).str.strip()
    lai_col = 'annual_mean_lai' if 'annual_mean_lai' in df_lai.columns else (
              'lai' if 'lai' in df_lai.columns else 'mean')
    lai = pd.Series(index=years, dtype=float)
    for y in years:
        mask = (df_lai.year == y) & (df_lai.scenario.str.contains(SCEN, case=False))
        vals = df_lai.loc[mask, lai_col]
        lai[y] = vals.iloc[0] if not vals.empty else np.nan

    # TP
    df_tp = pd.read_csv(TP_CSV)
    df_tp['scenario'] = df_tp['scenario'].astype(str).str.strip()
    tp = pd.Series(index=years, dtype=float)
    for y in years:
        vals = df_tp.loc[(df_tp.year == y) & (df_tp.scenario == SCEN), 'tp']
        tp[y] = vals.iloc[0] if not vals.empty else np.nan

    # STL
    df_stl = pd.read_csv(STL_CSV)
    df_stl['scenario'] = df_stl['scenario'].astype(str).str.strip()
    stl = pd.Series(index=years, dtype=float)
    for y in years:
        vals = df_stl.loc[(df_stl.year == y) & (df_stl.scenario == SCEN), 'stl']
        stl[y] = vals.iloc[0] if not vals.empty else np.nan

    # DAM = annual mean of monthly sums of dam_rem_cap (ssp245)
    df_dam = pd.read_csv(DAM_REM_CSV)
    df_dam['scenario'] = df_dam['scenario'].astype(str).str.strip()
    dam = pd.Series(index=years, dtype=float)
    for y in years:
        vals = df_dam.loc[(df_dam.year == y) & (df_dam.scenario == SCEN), 'mean']
        dam[y] = vals.iloc[0] if not vals.empty else np.nan

    # quick availability check
    print("Data availability 2025–2100 (non-NA counts):")
    for name, s in [('SOC', soc), ('LAI', lai), ('TP', tp), ('DAM', dam), ('STL', stl)]:
        print(f"  {name:<3}: {s.count():>3} / {len(years)}")

    df = pd.DataFrame({'soc': soc, 'lai': lai, 'tp': tp, 'dam': dam, 'stl': stl})
    df = df.dropna(subset=['soc','lai','tp','dam','stl'])
    df.index.name = 'year'
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 4) Feature builders (same as 1950–2024 version)
# ─────────────────────────────────────────────────────────────────────────────
def singles(df, drivers):
    return df[drivers].copy()

def singles_plus_exact_k(df, drivers, k):
    X = singles(df, drivers)  # always include singles
    for combo in combinations(drivers, k):
        name = "x".join(combo)
        prod = np.prod([df[c] for c in combo], axis=0)
        X[name] = prod
    return X

def features_set1(df, drivers):  # singles + only 2-way
    return singles_plus_exact_k(df, drivers, 2)

def features_set2(df, drivers):  # singles + only 3-way
    return singles_plus_exact_k(df, drivers, 3)

def features_set3(df, drivers):  # singles + only 4-way
    return singles_plus_exact_k(df, drivers, 4)

# ─────────────────────────────────────────────────────────────────────────────
# 5) VIP calc (match 1950–2024: manual scale X; center y; PLS scale=False)
# ─────────────────────────────────────────────────────────────────────────────
def pls_vip_contrib(X, y, n_components=2):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.shape[0] < 3 or X.shape[1] == 0 or np.allclose(np.std(y), 0):
        return np.full(X.shape[1], np.nan)

    Xc = StandardScaler().fit_transform(X)
    yc = y - y.mean()

    A = min(n_components, Xc.shape[0] - 1, Xc.shape[1])
    if A < 1:
        return np.full(X.shape[1], np.nan)

    pls = PLSRegression(n_components=A, scale=False).fit(Xc, yc)
    W = pls.x_weights_            # (p, A)
    U = pls.y_scores_             # (n, A)
    SSY = np.sum(U**2, axis=0)    # (A,)
    total_SSY = SSY.sum()
    p = W.shape[0]
    if total_SSY <= 0:
        return np.full(p, np.nan)

    VIP = np.zeros(p)
    for i in range(p):
        s = 0.0
        for a in range(A):
            wa = W[:, a]
            wia = W[i, a]
            denom = np.dot(wa, wa)
            s += SSY[a] * (wia**2 / denom if denom > 0 else 0.0)
        VIP[i] = np.sqrt(p * s / total_SSY)

    sVIP = np.nansum(VIP)
    return (VIP / sVIP * 100) if sVIP > 0 else np.full(p, np.nan)

# ─────────────────────────────────────────────────────────────────────────────
# 6) MAIN: 1 CP → 2 segments (S1/S2) → sets 1/2/3 → wide layout (same format)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = load_series_2025_2100()
    if df.empty or len(df) < 6:
        print("Not enough complete rows after dropna() to run CP + PLS.")
        return

    yrs = df.index.values
    soc = df['soc'].values

    # single interior CP, then two segments
    cp_idx = sequential_mk_single_cp(soc)
    print(f"Detected CP at {int(yrs[cp_idx])} (idx {cp_idx} in 2025–2100)")

    segments = [
        ("S1", 0,          cp_idx),           # start .. CP
        ("S2", cp_idx + 1, len(yrs) - 1),     # CP+1 .. end
    ]

    drivers = ['lai', 'tp', 'dam', 'stl']
    builders = {
        'set1_2way': features_set1,
        'set2_3way': features_set2,
        'set3_4way': features_set3,
    }

    # collect (tidy)
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

    # wide layout: index = segment × segment_years × feature; columns = sets
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

    # column order to match 1950–2024 file
    cols = ['segment', 'segment_years', 'feature', 'set1_2way', 'set2_3way', 'set3_4way']
    wide = wide.reindex(columns=[c for c in cols if c in wide.columns])

    out_path = OUT_SUBDIR / "pls_interaction_vip_2025_2100.csv"
    wide.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    main()
