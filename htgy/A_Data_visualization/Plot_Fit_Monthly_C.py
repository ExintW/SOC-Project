# =============================================================================
# Monthly SOC Climatology (1950–2024): mean + 95% CI band + sine fit
# v5: remove std errorbars; show 95% confidence band instead
# =============================================================================
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, t  # correlation + t critical

# make OUTPUT_DIR available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # defines OUTPUT_DIR

# -----------------------------------------------------------------------------
# Font sizes (adjust here)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

# =============================================================================
# 1) Configuration & File-path Setup
# =============================================================================
past_nc     = OUTPUT_DIR / "Data" / "SOC_Past 2" / "Total_C_1950-2007_monthly.nc"
present_dir = OUTPUT_DIR / "Data" / "SOC_Present 7"
start_year  = 1950
end_year    = 2024  # inclusive

# =============================================================================
# 2) Helpers
# =============================================================================
def safe_read_present_parquet(parquet_path: Path) -> float:
    """
    Read one monthly Parquet file and return spatial mean of Total_C.
    Returns np.nan if file missing.
    """
    try:
        df = pd.read_parquet(parquet_path)
        return float(pd.to_numeric(df["Total_C"], errors="coerce").mean())
    except FileNotFoundError:
        return np.nan

def sin_model(m, A, phi, C):
    """y = A * sin(2*pi*m/12 + phi) + C, where m is month index."""
    return A * np.sin(2*np.pi * m/12.0 + phi) + C

def pretty_p(p):
    """Short pretty formatting for p-values in legend."""
    if p < 1e-99:
        return "<1e-99"
    return f"{p:.2e}"

def month_ci95(x: np.ndarray):
    """
    Given an array x (values across years for a fixed month),
    return (mean, lower95, upper95).
    Uses t-based CI: mean ± tcrit * s/sqrt(n)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        m = float(np.nanmean(x)) if n > 0 else np.nan
        return m, np.nan, np.nan

    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    tcrit = t.ppf(0.975, df=n - 1)
    margin = tcrit * s / np.sqrt(n)
    return m, m - margin, m + margin

# =============================================================================
# 3) Gather monthly spatial means (1950–2024)
# =============================================================================
records = []

# Past 1950–2006 from NetCDF
with xr.open_dataset(past_nc) as ds:
    ds_sel = ds.sel(time=slice(f"{start_year}-01-01", "2006-12-01"))
    for tt in ds_sel.time.values:
        arr = ds_sel["total_C"].sel(time=tt).values
        ts = pd.Timestamp(tt)
        records.append({
            "date":     ts,
            "year":     ts.year,
            "month":    ts.month,
            "soc_mean": float(np.nanmean(arr))
        })

# Present 2007–2024 from Parquet
for year in range(2007, end_year + 1):
    for month in range(1, 13):
        p = present_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"
        val = safe_read_present_parquet(p)
        if np.isfinite(val):
            records.append({
                "date":     pd.Timestamp(year=year, month=month, day=1),
                "year":     year,
                "month":    month,
                "soc_mean": float(val)
            })

df = pd.DataFrame(records)
df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

# =============================================================================
# 4) Monthly climatology across years: mean + 95% CI
# =============================================================================
rows = []
for m in range(1, 13):
    vals = df.loc[df["month"] == m, "soc_mean"].to_numpy(dtype=float)
    mm, lo, hi = month_ci95(vals)
    rows.append({
        "month": m,
        "month_mean": mm,
        "ci95_lower": lo,
        "ci95_upper": hi,
        "n_years": int(np.isfinite(vals).sum())
    })

clim = pd.DataFrame(rows).sort_values("month")

# =============================================================================
# 5) Fit sine: y = A sin(2π m/12 + φ) + C
# =============================================================================
months = clim["month"].values.astype(float)  # 1..12
y_obs  = clim["month_mean"].values

A0   = 0.5 * (np.nanmax(y_obs) - np.nanmin(y_obs)) if np.all(np.isfinite(y_obs)) else 1.0
phi0 = 0.0
C0   = np.nanmean(y_obs)

popt, pcov = curve_fit(sin_model, months, y_obs, p0=[A0, phi0, C0], maxfev=10000)
A_fit, phi_fit, C_fit = popt

y_pred = sin_model(months, A_fit, phi_fit, C_fit)

# R^2
ss_res = np.sum((y_obs - y_pred) ** 2)
ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
r2 = 1.0 - ss_res / (ss_tot + 1e-12)

# Pearson correlation r and p-value
r_pearson, p_value = pearsonr(y_obs, y_pred)

# Smooth curve
month_fine = np.linspace(0, 13, 520)
y_fit = sin_model(month_fine, A_fit, phi_fit, C_fit)

# =============================================================================
# 6) Save climatology table
# =============================================================================
clim_csv = OUTPUT_DIR / "monthly_soc_climatology_1950_2024_ci95.csv"
clim.to_csv(clim_csv, index=False)
print(f"Monthly SOC climatology (mean + 95% CI) saved to: {clim_csv}")

# =============================================================================
# 7) Plot: monthly mean points + 95% CI band + sine fit
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# 95% CI band
ax.fill_between(
    clim["month"],
    clim["ci95_lower"],
    clim["ci95_upper"],
    alpha=0.25,
    label="95% confidence band (1950–2024)"
)

# Monthly means (points)
ax.plot(
    clim["month"],
    clim["month_mean"],
    "o",
    markersize=5,
    label="Monthly mean (1950–2024)"
)

# Fit label with equation + R^2 + p
fit_label = (
    rf"Fit: $y={A_fit:.3f}\,\sin\!\left(\frac{{2\pi m}}{{12}}{phi_fit:+.3f}\right){C_fit:+.3f}$"
    "\n"
    rf"$R^2={r2:.3f}$, $p={pretty_p(p_value)}$"
)

# Sine fit line
ax.plot(month_fine, y_fit, linewidth=2.0, label=fit_label)

# axis & ticks
ax.set_xlim(0.5, 12.5)
ax.set_xticks(range(1, 13))
ax.set_xlabel("Month")
ax.set_ylabel("SOC (g/kg)")
ax.set_title("Monthly SOC Climatology Sinusoid Fit")
ax.grid(True, alpha=0.3)

ax.legend(loc="upper left", framealpha=0.9)

plt.tight_layout()
fig_out = OUTPUT_DIR / "monthly_soc_climatology_sinefit_1950_2024_CI95.png"
fig.savefig(fig_out, dpi=300)
print(f"Figure saved to: {fig_out}")

# =============================================================================
# 8) Console prints: fitted equation & wave height + R^2 & p-value
# =============================================================================
phi_deg = (np.degrees(phi_fit) + 180) % 360 - 180  # display in (-180, 180]
print("\n=== Sinusoid Fit (Monthly Climatology 1950–2024) ===")
print(f"y = {A_fit:.6g} * sin(2π*m/12 + {phi_fit:.6g}) + {C_fit:.6g}")
print(f"Amplitude A = {A_fit:.6g}")
print(f"Phase φ = {phi_fit:.6g} rad ({phi_deg:.3g}°)")
print(f"Mean level C = {C_fit:.6g}")
print(f"Wave height (peak-to-trough) = {2*abs(A_fit):.6g}")
print(f"R² = {r2:.6g}")
print(f"Pearson r = {r_pearson:.6g}")
print(f"p-value = {p_value:.6g}\n")
