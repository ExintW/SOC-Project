#!/usr/bin/env python3
import sys, os
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

# =============================================================================
# (1) CONFIGURATION & PATHS
# =============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from globals import PROCESSED_DIR, OUTPUT_DIR  # keep parity

OUT_SUBDIR = OUTPUT_DIR / 'Contribution_dam_rem'
OUT_SUBDIR.mkdir(exist_ok=True, parents=True)

# CSV paths
LAI_CSV = OUTPUT_DIR / 'annual_mean_lai_by_scenario.csv'
TP_CSV  = OUTPUT_DIR / 'tp_1950-2100_mean_tp.csv'
SOC_CSV = OUTPUT_DIR / 'annual_mean_soc_by_scenario.csv'
STL_CSV = OUTPUT_DIR / 'stl_1950-2100_mean_temperature.csv'
DAM_REM_CSV = OUTPUT_DIR / 'annual_mean_dam_rem_cap_by_scenario.csv'

years = np.arange(1950, 2025)  # 1950–2024 inclusive

# ================== Switches & Segment-specific config ==================
MAX_COMPONENTS     = 6
CONFIDENCE_Q2_MIN  = None   # e.g., 0.3 to mask low-confidence contributions

# Per-segment transform profile:
# - detrend: remove linear time trend in y and drivers
# - deltas : work on first-differences (Δ) to focus on change
# - lags   : add lai_lag1, dam_lag1 (then drop NA rows)
SEG_CFG = {
    'S1': {'detrend': True, 'deltas': True,  'lags': False},  # ↓ LAI credit from co-trend
    'S2': {'detrend': True, 'deltas': True,  'lags': False},
    'S3': {'detrend': True, 'deltas': True,  'lags': True},   # ↑ LAI/DAM via lagged effects
}

# Keep interactions; no PCA compression
BASE_DRIVERS = ['lai', 'tp', 'dam', 'stl']
# ========================================================================

# =============================================================================
# (2) MK break detection
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
    return np.argmin(diff[1:-1]) + 1

# =============================================================================
# (3) Load time series
# =============================================================================
def _mean_if_any(df, mask, col):
    return df.loc[mask, col].mean() if mask.any() else np.nan

def load_df():
    # SOC
    df = pd.read_csv(SOC_CSV)
    soc = pd.Series(index=years, dtype=float)
    for y in years:
        m = (df.year == y) & df.scenario.isin(['Past', 'Present'])
        soc[y] = _mean_if_any(df, m, 'mean')

    # LAI
    df = pd.read_csv(LAI_CSV); lai = pd.Series(index=years, dtype=float)
    for y in years:
        pat = 'Historical' if y <= 2000 else ('Present' if y <= 2014 else 'ssp245')
        m = (df.year == y) & df.scenario.str.contains(pat, case=False)
        lai[y] = _mean_if_any(df, m, 'annual_mean_lai')

    # TP
    df = pd.read_csv(TP_CSV); tp = pd.Series(index=years, dtype=float)
    for y in years:
        scen = 'Past' if y <= 2006 else 'Present'
        m = (df.year == y) & (df.scenario == scen)
        tp[y] = _mean_if_any(df, m, 'tp')

    # STL
    df = pd.read_csv(STL_CSV); stl = pd.Series(index=years, dtype=float)
    for y in years:
        scen = 'Past' if y <= 2006 else ('Present' if y <= 2014 else 'ssp245')
        m = (df.year == y) & (df.scenario == scen)
        stl[y] = _mean_if_any(df, m, 'stl')

    # DAM remained capacity
    df_dam = pd.read_csv(DAM_REM_CSV)
    dam = pd.Series(index=years, dtype=float)
    for y in years:
        scen = 'Past' if y <= 2006 else 'Present'
        m = (df_dam.year == y) & (df_dam.scenario == scen)
        dam[y] = _mean_if_any(df_dam, m, 'mean')

    df_all = pd.DataFrame({'soc': soc, 'lai': lai, 'tp': tp, 'dam': dam, 'stl': stl}).dropna()
    df_all.index.name = 'year'
    print(f"[load] kept {df_all.shape[0]}/{len(years)} years after NA filtering: {int(df_all.index.min())}–{int(df_all.index.max())}")
    return df_all

# =============================================================================
# (4) Segment transforms
# =============================================================================
def detrend_series(y, t=None):
    n = len(y)
    if t is None: t = np.arange(n)
    T = np.vstack([np.ones(n), t]).T
    coef, *_ = np.linalg.lstsq(T, y, rcond=None)
    return y - T @ coef

def detrend_df(df, cols):
    out = df.copy()
    t = np.arange(len(df))
    for c in cols:
        out[c] = detrend_series(df[c].astype(float).values, t)
    return out

def to_deltas(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = df[c].diff()
    return out

def add_lags(df, cols, lags=(1,)):
    out = df.copy()
    for c in cols:
        for L in lags:
            out[f"{c}_lag{L}"] = df[c].shift(L)
    return out

def apply_seg_transforms(seg_label, seg_df):
    cfg = SEG_CFG.get(seg_label, {'detrend':True, 'deltas':False, 'lags':False})
    if cfg.get('detrend', False):
        seg_df = detrend_df(seg_df, cols=['soc','lai','tp','dam','stl'])
    if cfg.get('deltas', False):
        seg_df = to_deltas(seg_df, cols=['soc','lai','tp','dam','stl'])
    if cfg.get('lags', False):
        seg_df = add_lags(seg_df, cols=['lai','dam'], lags=(1,))
    seg_df = seg_df.dropna()
    return seg_df, cfg

# =============================================================================
# (5) Feature builders (interactions allowed)
# =============================================================================
def singles(df, drivers):
    return df[drivers].copy()

def singles_plus_exact_k(df, drivers, k):
    X = df[drivers].copy()
    for combo in combinations(drivers, k):
        name = "x".join(combo)
        prod = df[combo[0]].copy()
        for c in combo[1:]:
            prod *= df[c]
        X[name] = prod
    return X

def features_set1(df, drivers):  # singles + 2-way
    return singles_plus_exact_k(df, drivers, 2)

def features_set2(df, drivers):  # singles + 3-way
    return singles_plus_exact_k(df, drivers, 3)

def features_set3(df, drivers):  # singles + 4-way
    return singles_plus_exact_k(df, drivers, 4)

BUILDERS = {
    'set1_2way': features_set1,
    'set2_3way': features_set2,
    'set3_4way': features_set3,
}

# =============================================================================
# (6) PLS with CV + metrics
# =============================================================================
def fit_pls_with_cv(X, y, max_comp=6):
    n, p = X.shape
    if n < 5 or p == 0 or np.allclose(np.std(y), 0, atol=1e-12):
        return None

    Xc = StandardScaler().fit_transform(X)
    yc = y - np.mean(y)

    Amax = min(max_comp, n - 1, p)
    if Amax < 1:
        return None

    n_splits = max(2, min(5, n // 2))
    kf = KFold(n_splits=n_splits, shuffle=False)

    best_A, best_q2 = 1, -np.inf
    for A in range(1, Amax + 1):
        pls = PLSRegression(n_components=A, scale=False)
        yhat_cv = cross_val_predict(pls, Xc, yc, cv=kf)
        q2 = r2_score(yc, yhat_cv)
        if q2 > best_q2:
            best_A, best_q2 = A, q2

    pls = PLSRegression(n_components=best_A, scale=False).fit(Xc, yc)
    yhat = pls.predict(Xc).ravel()
    r2y = r2_score(yc, yhat)
    return {"pls": pls, "Xc": Xc, "yc": yc, "A": best_A, "R2Y": r2y, "Q2": best_q2}

def vip_percent(pls, Xc, yc):
    W = pls.x_weights_
    U = pls.y_scores_
    p, A = W.shape
    SSY = np.sum(U**2, axis=0)
    total_SSY = SSY.sum()
    if total_SSY <= 1e-12:
        return np.full(p, np.nan)

    VIP = np.zeros(p)
    eps = 1e-12
    for i in range(p):
        s = 0.0
        for a in range(A):
            wa = W[:, a]
            denom = max(np.dot(wa, wa), eps)
            s += SSY[a] * (W[i, a]**2 / denom)
        VIP[i] = np.sqrt(p * s / total_SSY)

    sVIP = np.nansum(VIP)
    return (VIP / sVIP * 100.0) if sVIP > 1e-12 else np.full(p, np.nan)

def signed_coef_percent(pls):
    B = pls.coef_.ravel()
    if B.size == 0 or np.allclose(B, 0, atol=1e-12):
        return np.full(0, np.nan), B
    abs_sum = np.sum(np.abs(B))
    if abs_sum <= 1e-12:
        return np.full_like(B, np.nan), B
    signed_pct = (np.abs(B) / abs_sum * 100.0) * np.sign(B)
    return signed_pct, B

# =============================================================================
# (7) Aggregate contributions to base drivers (Shapley-style equal split)
# =============================================================================
def parse_factors(feat_name):
    # "laixtp" -> ["lai","tp"]; "dam" -> ["dam"]
    return feat_name.split('x')

def aggregate_to_drivers(rows_df, value_col, sets=('set1_2way','set2_3way','set3_4way')):
    agg_records = []
    for (seg, yrs) in rows_df[['segment','segment_years']].drop_duplicates().itertuples(index=False):
        for st in sets:
            sub = rows_df[(rows_df['segment']==seg) & (rows_df['segment_years']==yrs) & (rows_df['set']==st)]
            if sub.empty:
                continue
            totals = defaultdict(float)
            for _, r in sub.iterrows():
                name = r['feature']
                v = r[value_col]
                if pd.isna(v):
                    continue
                factors = [f for f in parse_factors(name) if f in BASE_DRIVERS]
                k = len(factors)
                if k == 0:
                    continue
                share = v / k
                for f in factors:
                    totals[f] += share
            for drv in BASE_DRIVERS:
                agg_records.append({
                    'segment': seg, 'segment_years': yrs,
                    'driver': drv, 'set': st, value_col: totals.get(drv, np.nan)
                })
    return pd.DataFrame(agg_records)

# =============================================================================
# (8) Main
# =============================================================================
def main():
    df = load_df()
    yrs = df.index.values
    soc_vals = df['soc'].values

    # locate CPs
    if not np.any(yrs <= 1995) or not np.any((yrs >= 1996) & (yrs <= 2005)):
        raise ValueError("Not enough data to locate two windows (<=1995 and 1996–2005).")
    i1 = np.where(yrs <= 1995)[0][-1]
    i2 = np.where(yrs <= 2005)[0][-1]

    cp1 = detect_one_mk_break(soc_vals[:i1 + 1])
    cp2 = detect_one_mk_break(soc_vals[i1 + 1:i2 + 1]) + (i1 + 1)
    print(f"Detected CP1={int(yrs[cp1])}, CP2={int(yrs[cp2])}")

    segments = [
        ("S1", 0,        cp1),
        ("S2", cp1 + 1,  cp2),
        ("S3", cp2 + 1,  len(yrs) - 1),
    ]

    tidy_rows = []
    diag_rows = []

    for seg_label, a, b in segments:
        if a > b:
            continue
        seg_df = df.iloc[a:b+1].copy()
        seg_years = f"{int(yrs[a])}-{int(yrs[b])}"

        # segment-specific transforms
        seg_df, cfg = apply_seg_transforms(seg_label, seg_df)
        if seg_df.shape[0] < 5:
            print(f"[warn] {seg_label} {seg_years}: too few rows after transforms ({seg_df.shape[0]}). Skipping.")
            continue

        drivers_seg = BASE_DRIVERS.copy()
        for set_name, builder in BUILDERS.items():
            X_df = builder(seg_df, drivers_seg)
            feat_names = list(X_df.columns)

            fit = fit_pls_with_cv(X_df.values, seg_df['soc'].values, max_comp=MAX_COMPONENTS)
            if fit is None:
                contrib_vip = np.full(len(feat_names), np.nan)
                contrib_signed = np.full(len(feat_names), np.nan)
                A = R2Y = Q2 = np.nan
            else:
                pls, Xc, yc, A, R2Y, Q2 = fit["pls"], fit["Xc"], fit["yc"], fit["A"], fit["R2Y"], fit["Q2"]
                contrib_vip = vip_percent(pls, Xc, yc)
                contrib_signed, _ = signed_coef_percent(pls)

            for name, v_vip, v_sig in zip(feat_names, contrib_vip, contrib_signed):
                tidy_rows.append({
                    'set': set_name,
                    'segment': seg_label,
                    'segment_years': seg_years,
                    'feature': name,
                    'vip_%': v_vip,
                    'signed_coef_%': v_sig
                })

            diag_rows.append({
                'set': set_name,
                'segment': seg_label,
                'segment_years': seg_years,
                'n_samples': int(seg_df.shape[0]),
                'n_features': int(X_df.shape[1]),
                'n_components': int(A) if not np.isnan(A) else np.nan,
                'R2Y_in_sample': float(R2Y) if not np.isnan(A) else np.nan,
                'Q2_cv': float(Q2) if not np.isnan(A) else np.nan,
                'transforms': f"detrend={cfg['detrend']}, deltas={cfg['deltas']}, lags={cfg['lags']}"
            })

    out_tidy = pd.DataFrame(tidy_rows)
    out_diag = pd.DataFrame(diag_rows)

    # Optional: mask low-confidence contributions
    if CONFIDENCE_Q2_MIN is not None:
        diag_key = out_diag.set_index(['set','segment','segment_years'])['Q2_cv']
        def _mask(df_vals, value_col):
            vals = []
            for _, r in df_vals.iterrows():
                q2 = diag_key.get((r['set'], r['segment'], r['segment_years']), np.nan)
                vals.append(r[value_col] if (not np.isnan(q2) and q2 >= CONFIDENCE_Q2_MIN) else np.nan)
            df_vals[value_col] = vals
            return df_vals
        out_tidy = _mask(out_tidy, 'vip_%')
        out_tidy = _mask(out_tidy, 'signed_coef_%')

    # Wide tables at feature level
    def to_wide(df_vals, value_col):
        wide = (
            df_vals
            .pivot_table(
                index=['segment', 'segment_years', 'feature'],
                columns='set',
                values=value_col,
                aggfunc='first'
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        desired = ['set1_2way','set2_3way','set3_4way']
        cols = ['segment','segment_years','feature'] + [c for c in desired if c in wide.columns]
        return wide.reindex(columns=cols)

    wide_vip    = to_wide(out_tidy, 'vip_%')
    wide_signed = to_wide(out_tidy, 'signed_coef_%')

    # === NEW: driver-level aggregation (splits interaction shares evenly) ===
    agg_vip    = aggregate_to_drivers(out_tidy, 'vip_%')
    agg_signed = aggregate_to_drivers(out_tidy, 'signed_coef_%')

    def to_wide_driver(df_vals, value_col):
        wide = (
            df_vals
            .pivot_table(
                index=['segment','segment_years','driver'],
                columns='set',
                values=value_col,
                aggfunc='first'
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        desired = ['set1_2way','set2_3way','set3_4way']
        cols = ['segment','segment_years','driver'] + [c for c in desired if c in wide.columns]
        return wide.reindex(columns=cols)

    wide_vip_drv    = to_wide_driver(agg_vip, 'vip_%')
    wide_signed_drv = to_wide_driver(agg_signed, 'signed_coef_%')

    # Export
    y0, y1 = int(df.index.min()), int(df.index.max())

    vip_path = OUT_SUBDIR / f'pls_interaction_vip_cv_transformed_{y0}_{y1}.csv'
    signed_path = OUT_SUBDIR / f'pls_interaction_signedcoef_cv_transformed_{y0}_{y1}.csv'
    diag_path = OUT_SUBDIR / f'pls_interaction_model_diag_transformed_{y0}_{y1}.csv'
    drv_vip_path = OUT_SUBDIR / f'driver_vip_share_cv_{y0}_{y1}.csv'
    drv_signed_path = OUT_SUBDIR / f'driver_signed_share_cv_{y0}_{y1}.csv'

    wide_vip.to_csv(vip_path, index=False)
    wide_signed.to_csv(signed_path, index=False)
    out_diag.to_csv(diag_path, index=False)
    wide_vip_drv.to_csv(drv_vip_path, index=False)
    wide_signed_drv.to_csv(drv_signed_path, index=False)

    print(f"Saved feature-level VIP    → {vip_path}")
    print(f"Saved feature-level Signed → {signed_path}")
    print(f"Saved diagnostics          → {diag_path}")
    print(f"Saved driver VIP shares    → {drv_vip_path}")
    print(f"Saved driver Signed shares → {drv_signed_path}")


if __name__ == '__main__':
    main()
