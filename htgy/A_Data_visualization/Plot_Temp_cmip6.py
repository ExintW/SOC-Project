import sys
import os
from pathlib import Path
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── PROJECT SETUP ────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR  # Path objects

# ─── FUNCTIONS ────────────────────────────────────────────────────────────────
def compute_cmip6_tsl(scenario_tag, year_start, year_end):
    """
    Compute annual mean TSL from pre‐resampled CMIP6 files.
    Assumes top‐5‐layer average is already baked in.
    """
    fn = f"resampled_tsl_top5layers_points_{scenario_tag}_interp.nc"
    ds = nc.Dataset(PROCESSED_DIR / "CMIP6_Data_Monthly_Resampled" / fn)
    tsl = ds.variables['tsl'][:]    # (time, n_points)
    ds.close()

    # Figure out the file’s start year
    file_start = 1930 if scenario_tag == "hist" else 2015

    # offset into the array
    offset = (year_start - file_start) * 12
    nyears = year_end - year_start + 1

    years = list(range(year_start, year_end + 1))
    vals = []
    for i in range(nyears):
        i0 = offset + i*12
        i1 = i0 + 12
        # slice one full year of monthly data
        arr = tsl[i0:i1, :]          # shape (12, n_points)
        arr_mean = arr.mean(axis=0)  # mean over months → (n_points,)
        vals.append(arr_mean.mean()) # mean over points
    return years, vals

# ─── MAIN SCRIPT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Historical CMIP6 (1930–2014), but we only want 1950–2014
    hist_years, hist_stl = compute_cmip6_tsl("hist", 1950, 2014)
    last_hist_year, last_hist_val = hist_years[-1], hist_stl[-1]

    # 2) Future SSPs 2015–2100
    future_tags = {
        "126": "ssp126",
        "245": "ssp245",
        "370": "ssp370",
        "585": "ssp585"
    }
    cmip6_data = {}
    for tag, name in future_tags.items():
        yrs, vals = compute_cmip6_tsl(tag, 2015, 2100)
        cmip6_data[tag] = {"name": name, "years": yrs, "stl": vals}

    # 3) Build long‐format DataFrame with labels Past/Present then SSPs
    records = []
    for yr, val in zip(hist_years, hist_stl):
        phase = "Past" if yr <= 2006 else "Present"
        records.append({"scenario": phase, "year": yr, "stl": val})
    for info in cmip6_data.values():
        for yr, val in zip(info["years"], info["stl"]):
            records.append({"scenario": info["name"], "year": yr, "stl": val})

    df_long = pd.DataFrame(records)

    # 4) Save CSV
    out_csv = OUTPUT_DIR / "stl_1950-2100_mean_temperature.csv"
    df_long.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved STL CSV to: {out_csv}")

    # 5) Plot time series
    plt.figure(figsize=(10, 6))

    # Past (1950–2006)
    yrs_past = [y for y in hist_years if y <= 2006]
    vals_past = [v for y,v in zip(hist_years, hist_stl) if y <= 2006]
    plt.plot(yrs_past, vals_past,
             color="black", linewidth=1, label="Past (1950–2006)")

    # Present (2007–2014)
    yrs_pres = [y for y in hist_years if y > 2006]
    vals_pres = [v for y,v in zip(hist_years, hist_stl) if y > 2006]
    plt.plot(yrs_pres, vals_pres,
             color="gray", linewidth=1, label="Present (2007–2014)")

    # SSPs (stitch to 2014)
    for tag, info in cmip6_data.items():
        yrs_plot = [last_hist_year] + info["years"]
        stl_plot = [last_hist_val] + info["stl"]
        plt.plot(yrs_plot, stl_plot,
                 linewidth=1, label=info["name"])

    plt.xlabel("Year")
    plt.ylabel("Mean Soil Temperature (K)")
    plt.title("Soil Temperature 1950–2100: CMIP6 Past / Present + SSP Scenarios")
    plt.legend()
    plt.tight_layout()

    out_png = OUTPUT_DIR / "stl_1950-2100_cmip6_scenarios.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved plot to: {out_png}")
    plt.show()
