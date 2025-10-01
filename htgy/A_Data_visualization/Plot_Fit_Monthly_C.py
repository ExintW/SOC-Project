# =============================================================================
# Monthly SOC Climatology (1950–2024): mean ± std per month + sine fit (v2)
# =============================================================================
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# make OUTPUT_DIR available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # defines OUTPUT_DIR

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
    try:
        df = pd.read_parquet(parquet_path)
        return float(df["Total_C"].mean())
    except FileNotFoundError:
        return np.nan

def sin_model(m, A, phi, C):
    # m is month index (1..12 normally)
    return A * np.sin(2*np.pi * m/12.0 + phi) + C

# =============================================================================
# 3) Gather monthly spatial means (1950–2024)
# =============================================================================
records = []

# Past 1950–2006 from NetCDF
with xr.open_dataset(past_nc) as ds:
    ds_sel = ds.sel(time=slice(f"{start_year}-01-01", "2006-12-01"))
    for t in ds_sel.time.values:
        arr = ds_sel["total_C"].sel(time=t).values
        records.append({
            "date":   pd.Timestamp(t),
            "year":   pd.Timestamp(t).year,
            "month":  pd.Timestamp(t).month,
            "soc_mean": np.nanmean(arr)
        })

# Present 2007–2024 from Parquet
for year in range(2007, end_year+1):
    for month in range(1, 13):
        p = present_dir / f"SOC_terms_{year}_{month:02d}_River.parquet"
        val = safe_read_present_parquet(p)
        if np.isfinite(val):
            records.append({
                "date":   pd.Timestamp(year=year, month=month, day=1),
                "year":   year,
                "month":  month,
                "soc_mean": val
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
months = clim["month"].values.astype(float)     # 1..12
y_obs  = clim["month_mean"].values

A0   = 0.5 * (np.nanmax(y_obs) - np.nanmin(y_obs)) if np.all(np.isfinite(y_obs)) else 1.0
phi0 = 0.0
C0   = np.nanmean(y_obs)

popt, pcov = curve_fit(sin_model, months, y_obs, p0=[A0, phi0, C0], maxfev=10000)
A_fit, phi_fit, C_fit = popt
phi_deg = (np.degrees(phi_fit) + 180) % 360 - 180  # display in (-180, 180]

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
# 7) Plot with error bars + sine fit + on-figure equation
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5))

# error bars (±1σ)
ax.errorbar(
    clim["month"], clim["month_mean"], yerr=clim["month_std"],
    fmt="o", capsize=4, linewidth=1.0, markersize=5, label="Monthly mean ± 1σ (1950–2024)"
)

# fitted sine
ax.plot(month_fine, y_fit, linewidth=2.0, label="Sine fit: A·sin(2πm/12 + φ) + C")

# axis & ticks (pad so months 1 and 12 are fully visible)
ax.set_xlim(0.5, 12.5)
ax.set_xticks(range(1, 13))
ax.set_xlabel("Month")
ax.set_ylabel("SOC (Total_C) — spatial mean")
ax.set_title("Monthly SOC Climatology (1950–2024) with Sinusoid Fit")
ax.grid(True, alpha=0.3)
ax.legend()

# equation box on-figure
eq_text = (
    f"Fit: y = {A_fit:.3f}·sin(2πm/12 + {phi_fit:.3f}) + {C_fit:.3f}\n"
    f"A = {A_fit:.3f},  φ = {phi_fit:.3f} rad ({phi_deg:.1f}°),  C = {C_fit:.3f}\n"
    f"Wave height (peak–trough) = {2*abs(A_fit):.3f}"
)
ax.text(0.02, 0.98, eq_text, transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round", alpha=0.1, pad=0.5))

plt.tight_layout()
fig_out = OUTPUT_DIR / "monthly_soc_climatology_sinefit_1950_2024_v2.png"
fig.savefig(fig_out, dpi=300)
print(f"Figure saved to: {fig_out}")

# =============================================================================
# 8) Console prints: fitted equation & wave height
# =============================================================================
print("\n=== Sinusoid Fit (Monthly Climatology 1950–2024) ===")
print(f"y = {A_fit:.6g} * sin(2π*m/12 + {phi_fit:.6g}) + {C_fit:.6g}")
print(f"Amplitude A = {A_fit:.6g}")
print(f"Phase φ = {phi_fit:.6g} rad ({phi_deg:.3g}°)")
print(f"Mean level C = {C_fit:.6g}")
print(f"Wave height (peak-to-trough) = {2*abs(A_fit):.6g}\n")
