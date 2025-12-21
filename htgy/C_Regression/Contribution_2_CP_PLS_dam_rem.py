#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PLS VIP with mechanism-aware transforms and a stronger S3-only STL adjustment.

Output tweak:
- Pretty feature names (LAI, DAM, TP, STL) with interaction joiner (default '×').
- CSVs are written with encoding='utf-8-sig' so Excel renders Unicode '×' correctly.

Core (baseline kept):
- Robust CSV extraction (mean over matching rows)
- DAM/LAI: 3-yr MA + memory
- TP: fixed-baseline anomaly (1950–1990)
- STL: 25-yr rolling anomaly (fallback fixed baseline)
- Global scaling on singles
- Asymmetric residualization (robust ridge)
- Hierarchical orthogonalized interactions
- CPs from SOC
- VIP% with CV-selected PLS components
- S3-only strong STL adjustment (trend¹–³, low-freq 11/21y, AR(1), TP level+resid)
"""

import sys, os, re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# =============================================================================
# (1) CONFIGURATION & PATHS
# =============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from globals import PROCESSED_DIR, OUTPUT_DIR  # expected in your project

OUT_SUBDIR = OUTPUT_DIR / 'Contribution_dam_rem'
OUT_SUBDIR.mkdir(exist_ok=True, parents=True)

LAI_CSV = OUTPUT_DIR / 'annual_mean_lai_by_scenario.csv'
TP_CSV  = OUTPUT_DIR / 'tp_1950-2100_mean_tp.csv'
SOC_CSV = OUTPUT_DIR / 'annual_mean_soc_by_scenario.csv'
STL_CSV = OUTPUT_DIR / 'stl_1950-2100_mean_temperature.csv'
DAM_REM_CSV = OUTPUT_DIR / 'annual_mean_dam_rem_cap_by_scenario.csv'

years = np.arange(1950, 2025)  # 1950–2024

# toggle S3 STL adjustment
ADJUST_S3_STL = True

# ---- CSV/pretty naming options ----
FEATURE_JOIN = '×'          # change to '*' or 'x' if you want ASCII-only
CSV_ENCODING = 'utf-8-sig'  # BOM so Excel recognizes UTF-8 and shows '×'

# =============================================================================
# (2) MK break detection
# =============================================================================
def sequential_mk(x: np.ndarray):
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

def detect_one_mk_break(x: np.ndarray) -> int:
    UF, UB = sequential_mk(x)
    diff = np.abs(UF - UB)
    return int(np.argmin(diff[1:-1]) + 1)

# =============================================================================
# (3) Helpers
# =============================================================================
def pick_mean(df: pd.DataFrame, mask, col: str) -> float:
    """Return numeric mean of df.loc[mask, col], or np.nan."""
    vals = pd.to_numeric(df.loc[mask, col], errors='coerce')
    if vals.empty:
        return float('nan')
    return float(vals.mean())

# ---- Pretty-printing of feature names for CSV output ----
TOKEN_MAP = {
    'lai_ma3_z': 'LAI',
    'dam_ma3_z': 'DAM',
    'tp_anom_z_resid': 'TP',
    'stl_anom_z_resid': 'STL',
    # fallbacks if any non-resid singles slip through
    'tp_anom_z': 'TP',
    'stl_anom_z': 'STL',
    'lai_ma3': 'LAI',
    'dam_ma3': 'DAM',
    'tp_anom': 'TP',
    'stl_anom': 'STL',
}

def _clean_token(tok: str) -> str:
    if tok in TOKEN_MAP:
        return TOKEN_MAP[tok]
    t = tok
    # strip common suffixes
    t = re.sub(r'_ma\d+', '', t)      # _ma3, _ma5, etc.
    t = t.replace('_anom', '')
    t = t.replace('_resid', '')
    t = t.replace('_z', '')
    # canonicalize base names to upper
    t = t.replace('lai', 'LAI').replace('dam', 'DAM').replace('tp', 'TP').replace('stl', 'STL')
    for base in ('LAI', 'DAM', 'TP', 'STL'):
        if t.upper() == base:
            return base
    return t.upper()

def pretty_feature(raw: str) -> str:
    """Map 'lai_ma3_zxstl_anom_z_resid' -> 'LAI×STL' (joiner set by FEATURE_JOIN)."""
    parts = raw.split('x')
    cleaned = [_clean_token(p) for p in parts]
    return FEATURE_JOIN.join(cleaned)

# =============================================================================
# (4) Load raw time series (robust)
# =============================================================================
def load_df_raw() -> pd.DataFrame:
    # SOC
    df = pd.read_csv(SOC_CSV)
    soc = pd.Series(index=years, dtype=float)
    for y in years:
        m = (df['year'] == y) & df['scenario'].isin(['Past', 'Present'])
        soc[y] = pick_mean(df, m, 'mean')

    # LAI
    df = pd.read_csv(LAI_CSV)
    lai = pd.Series(index=years, dtype=float)
    for y in years:
        pat = 'Historical' if y <= 2000 else ('Present' if y <= 2014 else 'ssp245')
        scen_mask = df['scenario'].astype(str).str.contains(pat, case=False, na=False)
        m = (df['year'] == y) & scen_mask
        lai[y] = pick_mean(df, m, 'annual_mean_lai')

    # TP (Past ≤2006; Present ≥2007)
    df = pd.read_csv(TP_CSV)
    tp = pd.Series(index=years, dtype=float)
    for y in years:
        scen = 'Past' if y <= 2006 else 'Present'
        m = (df['year'] == y) & (df['scenario'] == scen)
        tp[y] = pick_mean(df, m, 'tp')

    # STL (Past ≤2006; Present 2007–2014; ssp245 ≥2015)
    df = pd.read_csv(STL_CSV)
    stl = pd.Series(index=years, dtype=float)
    for y in years:
        scen = 'Past' if y <= 2006 else ('Present' if y <= 2014 else 'ssp245')
        m = (df['year'] == y) & (df['scenario'] == scen)
        stl[y] = pick_mean(df, m, 'stl')

    # DAM remained capacity (Past ≤2006; Present ≥2007)
    df_dam = pd.read_csv(DAM_REM_CSV)
    dam = pd.Series(index=years, dtype=float)
    for y in years:
        scen = 'Past' if y <= 2006 else 'Present'
        m = (df_dam['year'] == y) & (df_dam['scenario'] == scen)
        dam[y] = pick_mean(df_dam, m, 'mean')

    df_all = pd.DataFrame({'soc': soc, 'lai': lai, 'tp': tp, 'dam': dam, 'stl': stl})
    df_all.index.name = 'year'
    return df_all

# =============================================================================
# (5) Transforms + scaling + robust residualization
# =============================================================================
def fixed_anomaly(series: pd.Series, ref_start=1950, ref_end=1990):
    ref = series.loc[(series.index >= ref_start) & (series.index <= ref_end)].mean()
    return series - ref

def rolling_anomaly(series: pd.Series, win=25, center=True, min_periods=8):
    return series - series.rolling(window=win, center=center, min_periods=min_periods).mean()

def _z(s: pd.Series) -> pd.Series:
    v = s.dropna()
    mu, sd = v.mean(), v.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.where(np.isfinite(s), 0.0, np.nan), index=s.index)
    return (s - mu) / sd

def residualize_against_safe(df: pd.DataFrame, target: str, predictors: list, ridge: float = 1e-6) -> pd.Series:
    y_all = df[target].values.astype(float)
    X_all = df[predictors].values.astype(float)
    mask = np.isfinite(y_all)
    for j in range(X_all.shape[1]):
        mask &= np.isfinite(X_all[:, j])
    resid = np.full_like(y_all, np.nan, dtype=float)
    n_eff = int(mask.sum())
    if n_eff < len(predictors) + 2:
        return pd.Series(resid, index=df.index)
    X = X_all[mask]; y = y_all[mask]
    X = np.column_stack([np.ones(len(X)), X])  # intercept
    XT_X = X.T @ X
    I = np.eye(XT_X.shape[0]); I[0,0] = 0.0
    try:
        beta = np.linalg.solve(XT_X + ridge*I, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XT_X + ridge*I) @ (X.T @ y)
    resid_masked = y - (X @ beta)
    resid[mask] = resid_masked
    return pd.Series(resid, index=df.index, name=f"{target}_resid")

def add_transforms_global_scaling(df_all: pd.DataFrame):
    """
    Returns df_model with drivers and helper columns:
    ['soc','lai_ma3_z','dam_ma3_z','tp_anom_z_resid','stl_anom_z_resid',
     'stl_anom_z','tp_anom_z','year_z']
    """
    df = df_all.sort_index().copy()

    # Structure signals
    df['dam_ma3'] = df['dam'].rolling(window=3, min_periods=1).mean()
    df['lai_ma3'] = df['lai'].rolling(window=3, min_periods=1).mean()
    df['dam_ma5']    = df['dam'].rolling(window=5, min_periods=1).mean()
    df['lai_ma5']    = df['lai'].rolling(window=5, min_periods=1).mean()
    df['dam_ma3_l1'] = df['dam_ma3'].shift(1)
    df['lai_ma3_l1'] = df['lai_ma3'].shift(1)

    # Meteorology anomalies
    df['tp_anom']  = fixed_anomaly(df['tp'],  ref_start=1950, ref_end=1990)
    stl_roll       = rolling_anomaly(df['stl'], win=25, center=True, min_periods=8)
    stl_fb         = fixed_anomaly(df['stl'], ref_start=1950, ref_end=1980)
    df['stl_anom'] = stl_roll.fillna(stl_fb)

    # Global scaling on primary singles
    singles = df[['lai_ma3', 'tp_anom', 'dam_ma3', 'stl_anom']]
    valid = singles.dropna()
    if len(valid) == 0:
        raise ValueError("No valid rows to fit global scaler on transformed singles.")
    scaler_global = StandardScaler().fit(valid.values)
    Z = pd.DataFrame(
        scaler_global.transform(singles.values),
        index=singles.index,
        columns=[c + '_z' for c in singles.columns]
    )
    df = pd.concat([df, Z], axis=1)

    # z-score memory terms
    for col in ['dam_ma5','lai_ma5','dam_ma3_l1','lai_ma3_l1']:
        df[col + '_z'] = _z(df[col])

    # time basis (for S3 adjust)
    year_series = pd.Series(df.index.values, index=df.index)
    df['year_z']  = _z(year_series)

    # Asymmetric residualization with memory (robust)
    df['tp_anom_z_resid']  = residualize_against_safe(
        df, 'tp_anom_z',
        ['dam_ma3_z','lai_ma3_z','dam_ma5_z','lai_ma5_z','dam_ma3_l1_z','lai_ma3_l1_z','stl_anom_z'],
        ridge=1e-6
    )
    df['stl_anom_z_resid'] = residualize_against_safe(
        df, 'stl_anom_z',
        ['dam_ma3_z','lai_ma3_z','dam_ma5_z','lai_ma5_z','dam_ma3_l1_z','lai_ma3_l1_z'],
        ridge=1e-6
    )

    use_cols = ['soc','lai_ma3_z','dam_ma3_z','tp_anom_z_resid','stl_anom_z_resid',
                'stl_anom_z','tp_anom_z','year_z']
    df_model = df[use_cols].dropna()
    df_model.index.name = 'year'
    return df_model, scaler_global

# =============================================================================
# (6) Hierarchical orthogonalized features
# =============================================================================
def _project_out(D: np.ndarray, v: np.ndarray) -> np.ndarray:
    if D.ndim == 1: D = D[:,None]
    D1 = np.column_stack([np.ones(D.shape[0]), D])
    beta, *_ = np.linalg.lstsq(D1, v, rcond=None)
    return v - (D1 @ beta)

def build_orthogonal_design(df: pd.DataFrame, drivers: list, max_order: int) -> pd.DataFrame:
    X = df[drivers].copy()
    current = X.values
    for k in range(2, max_order+1):
        for combo in combinations(drivers, k):
            name = "x".join(combo)
            v = np.prod([df[c].values for c in combo], axis=0)
            X[name] = _project_out(current, v)
        current = X.values
    return X

def features_set1(df, drivers): return build_orthogonal_design(df, drivers, 2)
def features_set2(df, drivers): return build_orthogonal_design(df, drivers, 3)
def features_set3(df, drivers): return build_orthogonal_design(df, drivers, 4)

# =============================================================================
# (7) PLS with CV-selected components + VIP
# =============================================================================
def _choose_ncomp_cv(X: np.ndarray, y: np.ndarray, Amax: int = 3) -> int:
    n, p = X.shape
    if n < 4 or p == 0 or np.allclose(np.std(y),0): return 1
    Amax_eff = max(1, min(Amax, n-1, p))
    n_splits = n if n <= 8 else 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    best_A, best_score = 1, -np.inf
    for A in range(1, Amax_eff+1):
        pls = PLSRegression(n_components=A, scale=False)
        scores = []
        for tr, te in kf.split(X):
            pls.fit(X[tr], y[tr])
            yhat = pls.predict(X[te]).ravel()
            scores.append(r2_score(y[te], yhat))
        m = np.nanmean(scores)
        if m > best_score: best_score, best_A = m, A
    return best_A

def pls_vip_contrib_cv_noscale(X: np.ndarray, y: np.ndarray, Amax: int = 3) -> np.ndarray:
    if X.shape[0] < 3 or X.shape[1] == 0 or np.allclose(np.std(y),0):
        return np.full(X.shape[1], np.nan)
    yc = y - np.mean(y)
    A = _choose_ncomp_cv(X, yc, Amax=Amax)
    pls = PLSRegression(n_components=A, scale=False).fit(X, yc)
    W, U = pls.x_weights_, pls.y_scores_
    SSY = np.sum(U**2, axis=0); total = SSY.sum()
    p = W.shape[0]
    if total <= 0: return np.full(p, np.nan)
    VIP = np.zeros(p)
    for i in range(p):
        s = 0.0
        for a in range(A):
            wa = W[:,a]; denom = np.dot(wa,wa)
            s += SSY[a] * (W[i,a]**2 / denom if denom>0 else 0.0)
        VIP[i] = np.sqrt(p * s / total)
    sVIP = np.nansum(VIP)
    return (VIP / sVIP * 100.0) if sVIP>0 else np.full(p, np.nan)

# =============================================================================
# (8) Main
# =============================================================================
def main():
    df_raw = load_df_raw()
    df, _ = add_transforms_global_scaling(df_raw)

    yrs = df.index.values
    soc = df['soc'].values

    # CPs from SOC
    if not np.any(yrs <= 1995) or not np.any((yrs >= 1996) & (yrs <= 2005)):
        raise ValueError("Not enough data to locate two windows (<=1995 and 1996–2005).")
    i1 = np.where(yrs <= 1995)[0][-1]
    i2 = np.where(yrs <= 2005)[0][-1]
    cp1 = detect_one_mk_break(soc[:i1+1])
    cp2 = detect_one_mk_break(soc[i1+1:i2+1]) + (i1 + 1)
    print(f"Detected CP1={int(yrs[cp1])}, CP2={int(yrs[cp2])} (from SOC)")

    segments = [("S1", 0, cp1), ("S2", cp1+1, cp2), ("S3", cp2+1, len(yrs)-1)]

    # ---- Strong S3-only STL adjustment ----
    if ADJUST_S3_STL:
        _, a3, b3 = segments[2]
        s3 = df.iloc[a3:b3+1].copy()

        # time basis
        t  = s3['year_z'].values
        t2 = (t**2 - np.mean(t**2)) / (np.std(t**2) + 1e-12)
        t3 = (t**3 - np.mean(t**3)) / (np.std(t**3) + 1e-12)
        s3['year2_z'] = t2
        s3['year3_z'] = t3

        # STL low-frequency (11y & 21y centered rolling means within S3)
        stl_low11 = s3['stl_anom_z'].rolling(window=11, center=True, min_periods=4).mean()
        stl_low21 = s3['stl_anom_z'].rolling(window=21, center=True, min_periods=7).mean()
        s3['stl_low11_z'] = _z(stl_low11)
        s3['stl_low21_z'] = _z(stl_low21)

        # STL AR(1)
        s3['stl_lag1_z'] = _z(s3['stl_anom_z'].shift(1))

        predictors = [
            'dam_ma3_z','lai_ma3_z',
            'tp_anom_z_resid','tp_anom_z',
            'year_z','year2_z','year3_z',
            'stl_low11_z','stl_low21_z',
            'stl_lag1_z'
        ]

        s3['stl_anom_z_resid_adj'] = residualize_against_safe(
            s3, target='stl_anom_z', predictors=predictors, ridge=1e-6
        )

        df.iloc[a3:b3+1, df.columns.get_loc('stl_anom_z_resid')] = s3['stl_anom_z_resid_adj'].values
        print("Applied strong S3-only STL adjustment.")

    drivers = ['lai_ma3_z','dam_ma3_z','tp_anom_z_resid','stl_anom_z_resid']

    builders = {
        'set1_2way': features_set1,
        'set2_3way': features_set2,
        'set3_4way': features_set3,
    }

    tidy_rows = []
    for set_name, builder in builders.items():
        for seg_label, a, b in segments:
            if a > b: continue
            seg_df = df.iloc[a:b+1][['soc'] + drivers].dropna()
            if len(seg_df) < 3: continue
            X_df = builder(seg_df, drivers)
            feat_names = list(X_df.columns)
            contrib = pls_vip_contrib_cv_noscale(X_df.values, seg_df['soc'].values, Amax=3)
            seg_years = f"{int(yrs[a])}-{int(yrs[b])}"
            for name, v in zip(feat_names, contrib):
                tidy_rows.append({
                    'set': set_name,
                    'segment': seg_label,
                    'segment_years': seg_years,
                    'feature': pretty_feature(name),
                    'contrib_%': v
                })

    out_tidy = pd.DataFrame(tidy_rows)

    # Save tidy with pretty names (encoding with BOM so Excel sees UTF-8)
    tidy_path = OUT_SUBDIR / 'pls_interaction_vip_1950_2024_tidy.csv'
    out_tidy.to_csv(tidy_path, index=False, encoding=CSV_ENCODING)
    print(f"Saved → {tidy_path}")

    # Wide layout using pretty feature names
    wide = (out_tidy
        .pivot_table(index=['segment','segment_years','feature'],
                     columns='set', values='contrib_%', aggfunc='first')
        .reset_index().rename_axis(None, axis=1))
    cols = ['segment','segment_years','feature','set1_2way','set2_3way','set3_4way']
    wide = wide.reindex(columns=[c for c in cols if c in wide.columns])

    out_path = OUT_SUBDIR / 'pls_interaction_vip_1950_2024.csv'
    wide.to_csv(out_path, index=False, encoding=CSV_ENCODING)
    print(f"Saved → {out_path}")

if __name__ == '__main__':
    main()
