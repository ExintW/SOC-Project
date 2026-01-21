# =============================================================================
# Monthly SOC Climatology (1950–2024): mean ± std per month + sine fit
# v4: remove text box; legend shows fitted equation + R^2 + p-value
# =============================================================================
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr  # correlation r and p-value

# make OUTPUT_DIR available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # defines OUTPUT_DIR

# -----------------------------------------------------------------------------
# Font sizes (adjust here)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,          # base font (fallback)
    "axes.titlesize": 18,     # title
    "axes.labelsize": 18,     # x/y labels
    "legend.fontsize": 14,    # legend text (slightly smaller to fit equation)
    "xtick.labelsize": 18,    # x tick labels
    "ytick.labelsize": 18,    # y tick labels
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
        return float(df["Total_C"].mean())
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

# =============================================================================
# 3) Gather monthly spatial means (1950–2024)
# =============================================================================
records = []

# Past 1950–2006 from NetCDF
with xr.open_dataset(past_nc) as ds:
    ds_sel = ds.sel(time=slice(f"{start_year}-01-01", "2006-12-01"))
    # If your NetCDF variable name differs, change "total_C" here
    for t in ds_sel.time.values:
        arr = ds_sel["total_C"].sel(time=t).values
        ts = pd.Timestamp(t)
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
# 4) Monthly climatology across years: mean & std
# =============================================================================
clim = (
    df.groupby("month")["soc_mean"]
      .agg(month_mean="mean", month_std=lambda x: np.std(x, ddof=1))
      .reset_index()
      .sort_values("month")
)

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

# Fitted values at the 12 observed months
y_pred = sin_model(months, A_fit, phi_fit, C_fit)

# R^2 (residual definition)
ss_res = np.sum((y_obs - y_pred) ** 2)
ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
r2 = 1.0 - ss_res / (ss_tot + 1e-12)

# Pearson correlation r and p-value
r_pearson, p_value = pearsonr(y_obs, y_pred)

# Smooth curve (extend beyond 1..12 so edges aren’t cramped)
month_fine = np.linspace(0, 13, 520)
y_fit = sin_model(month_fine, A_fit, phi_fit, C_fit)

# =============================================================================
# 6) Save climatology table
# =============================================================================
clim_csv = OUTPUT_DIR / "monthly_soc_climatology_1950_2024.csv"
clim.to_csv(clim_csv, index=False)
print(f"Monthly SOC climatology saved to: {clim_csv}")

# =============================================================================
# 7) Plot with error bars + sine fit (NO text box; legend contains equation + stats)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# error bars (±1σ)
ax.errorbar(
    clim["month"], clim["month_mean"], yerr=clim["month_std"],
    fmt="o", capsize=4, linewidth=1.0, markersize=5,
    label="Monthly mean ± 1σ (1950–2024)"
)

# Legend label showing the fitted equation + R^2 + p
# Use mathtext so it looks like an equation in the legend.
fit_label = (
    rf"Fit: $y={A_fit:.3f}\,\sin\!\left(\frac{{2\pi m}}{{12}}{phi_fit:+.3f}\right){C_fit:+.3f}$"
    "\n"
    rf"$R^2={r2:.3f}$, $p={pretty_p(p_value)}$"
)

ax.plot(month_fine, y_fit, linewidth=2.0, label=fit_label)

# axis & ticks (pad so months 1 and 12 are fully visible)
ax.set_xlim(0.5, 12.5)
ax.set_xticks(range(1, 13))
ax.set_xlabel("Month")
ax.set_ylabel("SOC (g/kg)")
ax.set_title("Monthly SOC Climatology (1950–2024) with Sinusoid Fit")
ax.grid(True, alpha=0.3)

# If legend is too large, you can:
# - reduce legend.fontsize above
# - move it to a different corner
# - set ncol=1 or 2
ax.legend(loc="upper left", framealpha=0.9)

plt.tight_layout()
fig_out = OUTPUT_DIR / "monthly_soc_climatology_sinefit_1950_2024_v4.png"
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
