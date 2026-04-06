#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PLS VIP with mechanism-aware transforms and a stronger S3-only STL adjustment.

New additions in this version:
1) Extract PLS linear equation for each segment:
   SOC = intercept + sum(beta_i * feature_i)
2) Save coefficients to CSV:
   OUTPUT_DIR/Contribution_dam_rem/pls_segment_equations_coefficients.csv
3) Show a short equation summary in the top-right of the SOC time series plot

Notes:
- PLS is fit on centered y inside fit_final_pls_for_segment, so intercept must be computed on original SOC scale.
- The plotted equation is shortened (top K terms by abs coefficient) to fit nicely.
"""

import sys, os, re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# =============================================================================
# (1) CONFIGURATION and PATHS
# =============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR  # expected in your project

OUT_SUBDIR = OUTPUT_DIR / "Contribution_dam_rem"
OUT_SUBDIR.mkdir(exist_ok=True, parents=True)

LAI_CSV = OUTPUT_DIR / "annual_mean_lai_by_scenario.csv"
TP_CSV  = OUTPUT_DIR / "tp_1950-2100_mean_tp.csv"
SOC_CSV = OUTPUT_DIR / "annual_mean_soc_by_scenario.csv"
STL_CSV = OUTPUT_DIR / "stl_1950-2100_mean_temperature.csv"
DAM_REM_CSV = OUTPUT_DIR / "annual_mean_dam_rem_cap_by_scenario.csv"

years = np.arange(1950, 2025)  # 1950 to 2024

# toggle S3 STL adjustment
ADJUST_S3_STL = True

# interaction scaling: <1.0 down-weights 2-way terms relative to mains
INTERACTION_SCALE = 0.7

# CSV and pretty naming options
FEATURE_JOIN = "×"          # change to "*" or "x" if you want ASCII-only
CSV_ENCODING = "utf-8-sig"  # BOM so Excel recognizes UTF-8 and shows "×"

# equation annotation options
EQUATION_TOPK_TERMS = 8     # how many terms to show on plot
EQUATION_DECIMALS = 3       # decimals shown in plot equation

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
    vals = pd.to_numeric(df.loc[mask, col], errors="coerce")
    if vals.empty:
        return float("nan")
    return float(vals.mean())

# Pretty-printing of feature names for CSV output
TOKEN_MAP = {
    "lai_ma3_z": "LAI",
    "dam_ma3_z": "DAM",
    "tp_anom_z_resid": "TP",
    "stl_anom_z_resid": "STL",
    # fallbacks if any non-resid singles slip through
    "tp_anom_z": "TP",
    "stl_anom_z": "STL",
    "lai_ma3": "LAI",
    "dam_ma3": "DAM",
    "tp_anom": "TP",
    "stl_anom": "STL",
}

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _clean_token(tok: str) -> str:
    if tok in TOKEN_MAP:
        return TOKEN_MAP[tok]
    t = tok
    t = re.sub(r"_ma\d+", "", t)      # _ma3, _ma5, etc.
    t = t.replace("_anom", "")
    t = t.replace("_resid", "")
    t = t.replace("_z", "")
    t = t.replace("lai", "LAI").replace("dam", "DAM").replace("tp", "TP").replace("stl", "STL")
    for base in ("LAI", "DAM", "TP", "STL"):
        if t.upper() == base:
            return base
    return t.upper()

def pretty_feature(raw: str) -> str:
    """Map 'lai_ma3_zxstl_anom_z_resid' to 'LAI×STL' (joiner set by FEATURE_JOIN)."""
    parts = raw.split("x")
    cleaned = [_clean_token(p) for p in parts]
    return FEATURE_JOIN.join(cleaned)

def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# =============================================================================
# (3b) Equation extraction helpers (NEW)
# =============================================================================
def extract_pls_equation(pls: PLSRegression,
                         feat_names: list,
                         y_mean_original: float):
    """
    Extract linear equation on ORIGINAL SOC scale.

    You fit PLS on centered y:
        yc = y - y_mean_original

    So:
        intercept_original = y_mean_original - (x_mean dot beta)

    Works across sklearn versions where x_mean is stored as:
      - pls.x_mean_   (newer)
      - pls._x_mean   (older)
    """
    beta = pls.coef_.ravel().astype(float)

    # sklearn version compatibility
    if hasattr(pls, "x_mean_"):
        x_mean = np.asarray(pls.x_mean_).ravel().astype(float)
    elif hasattr(pls, "_x_mean"):
        x_mean = np.asarray(pls._x_mean).ravel().astype(float)
    else:
        raise AttributeError(
            "PLSRegression does not have x_mean_ or _x_mean. "
            "Please check your sklearn version or pass X.mean(axis=0) manually."
        )

    intercept_original = float(y_mean_original) - float(np.dot(x_mean, beta))
    return intercept_original, beta


def format_equation_for_plot(intercept: float,
                             feat_names: list,
                             beta: np.ndarray,
                             topk: int = 6,
                             decimals: int = 3):
    """
    Make a short readable equation summary for plot annotation.
    Uses top K terms by abs(beta).
    """
    if beta is None or len(beta) == 0:
        return "Equation unavailable"

    order = np.argsort(np.abs(beta))[::-1]
    order = order[:min(topk, len(order))]

    lines = []
    lines.append(f"SOC = {intercept:.{decimals}f}")

    for i in order:
        sign = "+" if beta[i] >= 0 else "-"
        lines.append(f"  {sign} {abs(beta[i]):.{decimals}f} · {feat_names[i]}")

    return "\n".join(lines)

def equation_to_rows(seg_label: str,
                     seg_years_str: str,
                     intercept: float,
                     feat_names: list,
                     beta: np.ndarray):
    """
    Prepare rows for CSV: segment, years, intercept row, then each coefficient row.
    """
    rows = []
    rows.append({
        "segment": seg_label,
        "segment_years": seg_years_str,
        "term_type": "intercept",
        "feature": "INTERCEPT",
        "coefficient": float(intercept),
        "abs_coefficient": float(abs(intercept)),
    })
    for name, b in zip(feat_names, beta):
        rows.append({
            "segment": seg_label,
            "segment_years": seg_years_str,
            "term_type": "coefficient",
            "feature": name,
            "coefficient": float(b),
            "abs_coefficient": float(abs(b)),
        })
    return rows

# =============================================================================
# (4) Load raw time series (robust)
# =============================================================================
def load_df_raw() -> pd.DataFrame:
    # SOC
    df = pd.read_csv(SOC_CSV)
    soc = pd.Series(index=years, dtype=float)
    for y in years:
        m = (df["year"] == y) & df["scenario"].isin(["Past", "Present"])
        soc[y] = pick_mean(df, m, "mean")

    # LAI
    df = pd.read_csv(LAI_CSV)
    lai = pd.Series(index=years, dtype=float)
    for y in years:
        pat = "Historical" if y <= 2000 else ("Present" if y <= 2014 else "ssp245")
        scen_mask = df["scenario"].astype(str).str.contains(pat, case=False, na=False)
        m = (df["year"] == y) & scen_mask
        lai[y] = pick_mean(df, m, "annual_mean_lai")

    # TP (Past <= 2006; Present >= 2007)
    df = pd.read_csv(TP_CSV)
    tp = pd.Series(index=years, dtype=float)
    for y in years:
        scen = "Past" if y <= 2006 else "Present"
        m = (df["year"] == y) & (df["scenario"] == scen)
        tp[y] = pick_mean(df, m, "tp")

    # STL (Past <= 2006; Present 2007 to 2014; ssp245 >= 2015)
    df = pd.read_csv(STL_CSV)
    stl = pd.Series(index=years, dtype=float)
    for y in years:
        scen = "Past" if y <= 2006 else ("Present" if y <= 2014 else "ssp245")
        m = (df["year"] == y) & (df["scenario"] == scen)
        stl[y] = pick_mean(df, m, "stl")

    # DAM remained capacity (Past <= 2006; Present >= 2007)
    df_dam = pd.read_csv(DAM_REM_CSV)
    dam = pd.Series(index=years, dtype=float)
    for y in years:
        scen = "Past" if y <= 2006 else "Present"
        m = (df_dam["year"] == y) & (df_dam["scenario"] == scen)
        dam[y] = pick_mean(df_dam, m, "mean")

    df_all = pd.DataFrame({"soc": soc, "lai": lai, "tp": tp, "dam": dam, "stl": stl})
    df_all.index.name = "year"
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

    X = X_all[mask]
    y = y_all[mask]
    X = np.column_stack([np.ones(len(X)), X])  # intercept

    XT_X = X.T @ X
    I = np.eye(XT_X.shape[0])
    I[0, 0] = 0.0

    try:
        beta = np.linalg.solve(XT_X + ridge * I, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XT_X + ridge * I) @ (X.T @ y)

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
    df["dam_ma3"] = df["dam"].rolling(window=3, min_periods=1).mean()
    df["lai_ma3"] = df["lai"].rolling(window=3, min_periods=1).mean()
    df["dam_ma5"]    = df["dam"].rolling(window=5, min_periods=1).mean()
    df["lai_ma5"]    = df["lai"].rolling(window=5, min_periods=1).mean()
    df["dam_ma3_l1"] = df["dam_ma3"].shift(1)
    df["lai_ma3_l1"] = df["lai_ma3"].shift(1)

    # Meteorology anomalies
    df["tp_anom"] = fixed_anomaly(df["tp"], ref_start=1950, ref_end=1990)
    stl_roll = rolling_anomaly(df["stl"], win=25, center=True, min_periods=8)
    stl_fb   = fixed_anomaly(df["stl"], ref_start=1950, ref_end=1980)
    df["stl_anom"] = stl_roll.fillna(stl_fb)

    # Global scaling on primary singles
    singles = df[["lai_ma3", "tp_anom", "dam_ma3", "stl_anom"]]
    valid = singles.dropna()
    if len(valid) == 0:
        raise ValueError("No valid rows to fit global scaler on transformed singles.")
    scaler_global = StandardScaler().fit(valid.values)

    Z = pd.DataFrame(
        scaler_global.transform(singles.values),
        index=singles.index,
        columns=[c + "_z" for c in singles.columns]
    )
    df = pd.concat([df, Z], axis=1)

    # z-score memory terms
    for col in ["dam_ma5", "lai_ma5", "dam_ma3_l1", "lai_ma3_l1"]:
        df[col + "_z"] = _z(df[col])

    # time basis (for S3 adjust)
    year_series = pd.Series(df.index.values, index=df.index)
    df["year_z"] = _z(year_series)

    # Asymmetric residualization with memory
    df["tp_anom_z_resid"] = residualize_against_safe(
        df, "tp_anom_z",
        ["dam_ma3_z", "lai_ma3_z", "dam_ma5_z", "lai_ma5_z", "dam_ma3_l1_z", "lai_ma3_l1_z"],
        ridge=1e-6
    )
    df["stl_anom_z_resid"] = residualize_against_safe(
        df, "stl_anom_z",
        ["dam_ma3_z", "lai_ma3_z", "dam_ma5_z", "lai_ma5_z", "dam_ma3_l1_z", "lai_ma3_l1_z"],
        ridge=1e-6
    )

    use_cols = [
        "soc", "lai_ma3_z", "dam_ma3_z", "tp_anom_z_resid", "stl_anom_z_resid",
        "stl_anom_z", "tp_anom_z", "year_z"
    ]
    df_model = df[use_cols].dropna()
    df_model.index.name = "year"
    return df_model, scaler_global

# =============================================================================
# (6) Hierarchical orthogonalized 2-way features with scaling
# =============================================================================
def _project_out(D: np.ndarray, v: np.ndarray) -> np.ndarray:
    if D.ndim == 1:
        D = D[:, None]
    D1 = np.column_stack([np.ones(D.shape[0]), D])
    beta, *_ = np.linalg.lstsq(D1, v, rcond=None)
    return v - (D1 @ beta)

def build_orthogonal_design(df: pd.DataFrame, drivers: list,
                            max_order: int, inter_scale: float = 1.0) -> pd.DataFrame:
    """
    Build design with main effects and orthogonalized interactions.
    Interaction terms (k >= 2) are scaled by inter_scale.
    """
    X = df[drivers].copy()
    current = X.values
    for k in range(2, max_order + 1):
        for combo in combinations(drivers, k):
            name = "x".join(combo)
            v = np.prod([df[c].values for c in combo], axis=0)
            v_ortho = _project_out(current, v)
            if k >= 2:
                v_ortho = inter_scale * v_ortho
            X[name] = v_ortho
        current = X.values
    return X

def features_set1(df, drivers):
    """Singles plus 2-way interactions, interactions scaled by INTERACTION_SCALE."""
    return build_orthogonal_design(df, drivers, max_order=2, inter_scale=INTERACTION_SCALE)

# =============================================================================
# (7) PLS with CV-selected components + VIP
# =============================================================================
def _choose_ncomp_cv(X: np.ndarray, y: np.ndarray, Amax: int = 3) -> int:
    n, p = X.shape
    if n < 4 or p == 0 or np.allclose(np.std(y), 0):
        return 1
    Amax_eff = max(1, min(Amax, n - 1, p))
    n_splits = n if n <= 8 else 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    best_A, best_score = 1, -np.inf
    for A in range(1, Amax_eff + 1):
        pls = PLSRegression(n_components=A, scale=False)
        scores = []
        for tr, te in kf.split(X):
            pls.fit(X[tr], y[tr])
            yhat = pls.predict(X[te]).ravel()
            scores.append(r2_score(y[te], yhat))
        m = np.nanmean(scores)
        if m > best_score:
            best_score, best_A = m, A
    return best_A

def pls_vip_contrib_cv_noscale(X: np.ndarray, y: np.ndarray, Amax: int = 3) -> np.ndarray:
    if X.shape[0] < 3 or X.shape[1] == 0 or np.allclose(np.std(y), 0):
        return np.full(X.shape[1], np.nan)

    yc = y - np.mean(y)
    A = _choose_ncomp_cv(X, yc, Amax=Amax)
    pls = PLSRegression(n_components=A, scale=False).fit(X, yc)

    W = pls.x_weights_
    U = pls.y_scores_
    SSY = np.sum(U**2, axis=0)
    total = SSY.sum()
    p = W.shape[0]

    if total <= 0:
        return np.full(p, np.nan)

    VIP = np.zeros(p)
    for i in range(p):
        s = 0.0
        for a in range(A):
            wa = W[:, a]
            denom = np.dot(wa, wa)
            s += SSY[a] * (W[i, a]**2 / denom if denom > 0 else 0.0)
        VIP[i] = np.sqrt(p * s / total)

    sVIP = np.nansum(VIP)
    return (VIP / sVIP * 100.0) if sVIP > 0 else np.full(p, np.nan)

# =============================================================================
# (7b) PLS figures helpers
# =============================================================================
def choose_ncomp_cv_curve(X: np.ndarray, y: np.ndarray, Amax: int = 3):
    """
    Return A_list, mean_r2_list, best_A.
    """
    n, p = X.shape
    if n < 4 or p == 0 or np.allclose(np.std(y), 0):
        return [1], [np.nan], 1

    Amax_eff = max(1, min(Amax, n - 1, p))
    n_splits = n if n <= 8 else 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    A_list = []
    mean_r2_list = []
    best_A, best_score = 1, -np.inf

    for A in range(1, Amax_eff + 1):
        pls = PLSRegression(n_components=A, scale=False)
        scores = []
        for tr, te in kf.split(X):
            pls.fit(X[tr], y[tr])
            yhat = pls.predict(X[te]).ravel()
            scores.append(r2_score(y[te], yhat))
        m = float(np.nanmean(scores))
        A_list.append(A)
        mean_r2_list.append(m)
        if m > best_score:
            best_score, best_A = m, A

    return A_list, mean_r2_list, best_A

def fit_final_pls_for_segment(X: np.ndarray, y: np.ndarray, Amax: int = 3):
    """
    Fit final PLS on centered y, then return model and predictions on original y scale.
    """
    y_mean = float(np.mean(y))
    yc = y - y_mean

    A_list, mean_r2_list, best_A = choose_ncomp_cv_curve(X, yc, Amax=Amax)
    pls = PLSRegression(n_components=best_A, scale=False).fit(X, yc)

    yhat_c = pls.predict(X).ravel()
    yhat = yhat_c + y_mean

    return pls, yhat, best_A, (A_list, mean_r2_list), y_mean

def plot_pls_figures(seg_label: str,
                     seg_years: np.ndarray,
                     y: np.ndarray,
                     yhat: np.ndarray,
                     pls: PLSRegression,
                     feat_names: list,
                     vip: np.ndarray,
                     out_dir: Path,
                     equation_text: str = None,
                     plot_cfg: dict = None):
    """
    Save 4 figures per segment:
    1) observed vs predicted
    2) time series: observed and predicted (equation shown at top-right)
    3) scores plot (if n_components >= 2)
    4) VIP bar plot (Top 15)
    """
    _safe_mkdir(out_dir)
    cfg = (plot_cfg or {}).get(seg_label, {})

    rmse = compute_rmse(y, yhat)

    # 1) Observed vs Predicted
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.scatter(y, yhat, s=25)
    mn = float(np.nanmin([np.nanmin(y), np.nanmin(yhat)]))
    mx = float(np.nanmax([np.nanmax(y), np.nanmax(yhat)]))
    ax1.plot([mn, mx], [mn, mx], linewidth=1.0)
    ax1.set_title(f"{seg_label}: SOC model output vs PLS reconstruction (RMSE={rmse:.3f} g/kg)")
    ax1.set_xlabel("SOC (dynamic model output)")
    ax1.set_ylabel("SOC (PLS reconstruction)")
    fig1.tight_layout()
    fig1.savefig(out_dir / f"PLS_{seg_label}_obs_vs_pred.png", dpi=300)
    plt.close(fig1)

    # 2) Time series
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(seg_years, y, label="SOC Model Output", linewidth=1.2)
    ax2.plot(seg_years, yhat, label=f"PLS Reconstruction (RMSE={rmse:.3f} g/kg)", linewidth=1.2)

    ts_title = cfg.get("ts_title", f"{seg_label}: SOC time series")
    ax2.set_title(ts_title, fontsize=12, pad=20)

    if "ts_ylim" in cfg:
        ax2.set_ylim(cfg["ts_ylim"])

    ax2.set_xlabel("Year")
    ax2.set_ylabel("SOC (g/kg)")
    ax2.grid(True, alpha=0.3)

    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99), framealpha=0.9)

    equation_text = str(equation_text)
    equation_text = equation_text.replace("\n", " ").replace("\r", " ")
    equation_text = "".join(equation_text.split())

    if equation_text is not None and len(str(equation_text).strip()) > 0:
        ax2.text(
            0.01, 1.01,
            equation_text,
            transform=ax2.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            clip_on=False
        )

    fig2.subplots_adjust(top=0.80)
    fig2.savefig(out_dir / f"PLS_{seg_label}_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # 3) Scores plot (T1 vs T2)
    if hasattr(pls, "x_scores_") and pls.x_scores_ is not None and pls.x_scores_.shape[1] >= 2:
        T = pls.x_scores_
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.scatter(T[:, 0], T[:, 1], s=30)
        ax3.set_title(f"{seg_label}: Scores plot (T1 vs T2)")
        ax3.set_xlabel("T1")
        ax3.set_ylabel("T2")
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(out_dir / f"PLS_{seg_label}_scores_T1_T2.png", dpi=300)
        plt.close(fig3)

    # 4) VIP Top 15
    if vip is not None and len(vip) == len(feat_names):
        topk = min(15, len(vip))
        order = np.argsort(vip)[::-1][:topk]
        top_names = [feat_names[i] for i in order][::-1]
        top_vals = vip[order][::-1]

        fig4, ax4 = plt.subplots(figsize=(9, 6))
        ax4.barh(top_names, top_vals)

        ax4.set_title(f"{seg_label}: VIP contribution", fontsize=14, pad=12)
        ax4.set_xlabel("VIP contribution (%)", fontsize=12)
        ax4.set_ylabel("Feature", fontsize=12)

        ax4.tick_params(axis="x", labelsize=12, length=6, width=1.2)
        ax4.tick_params(axis="y", labelsize=12, length=0)

        fig4.tight_layout()
        fig4.savefig(out_dir / f"PLS_{seg_label}_VIP_top{topk}.png", dpi=300)
        plt.close(fig4)

# =============================================================================
# (8) Main
# =============================================================================
def main():
    df_raw = load_df_raw()
    df, _ = add_transforms_global_scaling(df_raw)

    yrs = df.index.values
    soc = df["soc"].values

    # CPs from SOC
    if not np.any(yrs <= 1995) or not np.any((yrs >= 1996) & (yrs <= 2005)):
        raise ValueError("Not enough data to locate two windows (<=1995 and 1996 to 2005).")

    i1 = np.where(yrs <= 1995)[0][-1]
    i2 = np.where(yrs <= 2005)[0][-1]
    cp1 = detect_one_mk_break(soc[:i1 + 1])
    cp2 = detect_one_mk_break(soc[i1 + 1:i2 + 1]) + (i1 + 1)
    print(f"Detected CP1={int(yrs[cp1])}, CP2={int(yrs[cp2])} (from SOC)")

    segments = [("S1", 0, cp1), ("S2", cp1 + 1, cp2), ("S3", cp2 + 1, len(yrs) - 1)]

    # Strong S3-only STL adjustment
    if ADJUST_S3_STL:
        _, a3, b3 = segments[2]
        s3 = df.iloc[a3:b3 + 1].copy()

        # time basis
        t = s3["year_z"].values
        t2 = (t**2 - np.mean(t**2)) / (np.std(t**2) + 1e-12)
        t3 = (t**3 - np.mean(t**3)) / (np.std(t**3) + 1e-12)
        s3["year2_z"] = t2
        s3["year3_z"] = t3

        # STL low-frequency within S3
        stl_low11 = s3["stl_anom_z"].rolling(window=11, center=True, min_periods=4).mean()
        stl_low21 = s3["stl_anom_z"].rolling(window=21, center=True, min_periods=7).mean()
        s3["stl_low11_z"] = _z(stl_low11)
        s3["stl_low21_z"] = _z(stl_low21)

        # STL AR(1)
        s3["stl_lag1_z"] = _z(s3["stl_anom_z"].shift(1))

        predictors = [
            "dam_ma3_z", "lai_ma3_z",
            "tp_anom_z_resid", "tp_anom_z",
            "year_z", "year2_z", "year3_z",
            "stl_low11_z", "stl_low21_z",
            "stl_lag1_z"
        ]

        s3["stl_anom_z_resid_adj"] = residualize_against_safe(
            s3, target="stl_anom_z", predictors=predictors, ridge=1e-6
        )

        df.iloc[a3:b3 + 1, df.columns.get_loc("stl_anom_z_resid")] = s3["stl_anom_z_resid_adj"].values
        print("Applied strong S3-only STL adjustment.")

    drivers = ["lai_ma3_z", "dam_ma3_z", "tp_anom_z_resid", "stl_anom_z_resid"]

    tidy_rows = []
    equation_rows = []  # NEW: store coefficients for CSV

    fig_dir = OUT_SUBDIR / "PLS_Figures"
    _safe_mkdir(fig_dir)

    for seg_label, a, b in segments:
        if a > b:
            continue

        seg_df = df.iloc[a:b + 1][["soc"] + drivers].dropna()
        if len(seg_df) < 3:
            continue

        # Design matrix
        X_df = features_set1(seg_df, drivers)
        feat_names_raw = list(X_df.columns)
        feat_names_pretty = [pretty_feature(n) for n in feat_names_raw]

        # VIP percent contribution
        contrib = pls_vip_contrib_cv_noscale(X_df.values, seg_df["soc"].values, Amax=3)

        # Fit final PLS for figures
        pls, yhat, best_A, cv_curve, y_mean_original = fit_final_pls_for_segment(
            X=X_df.values,
            y=seg_df["soc"].values,
            Amax=3
        )

        rmse = compute_rmse(seg_df["soc"].values, yhat)
        print(f"{seg_label}: chosen n_components = {best_A}, RMSE = {rmse:.3f} g/kg")

        # Extract equation (NEW)
        intercept, beta = extract_pls_equation(
            pls=pls,
            feat_names=feat_names_pretty,
            y_mean_original=y_mean_original
        )

        # Short equation string for plot annotation (NEW)
        eq_text = format_equation_for_plot(
            intercept=intercept,
            feat_names=feat_names_pretty,
            beta=beta,
            topk=EQUATION_TOPK_TERMS,
            decimals=EQUATION_DECIMALS
        )

        PLOT_CFG = {
            "S1": {
                "ts_title": "S1 (1950–1973): Early regime",
                "ts_ylim": (8.75, 10.8),

            },
            "S2": {
                "ts_title": "S2 (1974-2000): Transition regime",
                "ts_ylim": (9, 10.3),

            },
            "S3": {
                "ts_title": "S3 (2001–2024): Modern regime",
                "ts_ylim": (8.5, 13),
            }
        }

        # Save 4 figures with equation shown in time series plot
        plot_pls_figures(
            seg_label=seg_label,
            seg_years=seg_df.index.values.astype(int),
            y=seg_df["soc"].values,
            yhat=yhat,
            pls=pls,
            feat_names=feat_names_pretty,
            vip=contrib,
            out_dir=fig_dir,
            equation_text=eq_text,
            plot_cfg = PLOT_CFG
        )

        seg_years_str = f"{int(yrs[a])}-{int(yrs[b])}"

        # Store VIP tidy output
        for name, v in zip(feat_names_raw, contrib):
            tidy_rows.append({
                "segment": seg_label,
                "segment_years": seg_years_str,
                "feature": pretty_feature(name),
                "set1_2way": v
            })

        # Store equation coefficients (NEW)
        equation_rows.extend(
            equation_to_rows(
                seg_label=seg_label,
                seg_years_str=seg_years_str,
                intercept=intercept,
                feat_names=feat_names_pretty,
                beta=beta
            )
        )

    # Save VIP tidy with pretty names
    out_tidy = pd.DataFrame(tidy_rows)
    tidy_path = OUT_SUBDIR / "pls_interaction_vip_1950_2024_tidy.csv"
    out_tidy.to_csv(tidy_path, index=False, encoding=CSV_ENCODING)
    print(f"Saved -> {tidy_path}")

    out_path = OUT_SUBDIR / "pls_interaction_vip_1950_2024.csv"
    out_tidy.to_csv(out_path, index=False, encoding=CSV_ENCODING)
    print(f"Saved -> {out_path}")

    # Save equations CSV (NEW)
    eq_df = pd.DataFrame(equation_rows)
    eq_csv = OUT_SUBDIR / "pls_segment_equations_coefficients.csv"
    eq_df.to_csv(eq_csv, index=False, encoding=CSV_ENCODING)
    print(f"Saved segment equations -> {eq_csv}")

    print(f"Saved PLS figures to: {fig_dir}")

if __name__ == "__main__":
    main()
