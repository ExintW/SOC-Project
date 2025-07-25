import os
import sys
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Add project root to path so globals.py (which defines OUTPUT_DIR) can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import OUTPUT_DIR  # Expected to point at root Output folder
# ──────────────────────────────────────────────────────────────────────────────

# Scenario subfolder name (e.g. "126", "245", "585")
SCENARIO = "585"

def annual_summary(year: int, scenario: str = SCENARIO):
    """
    Compute annual summaries for a given year and scenario:
      - Read 12 monthly Parquet outputs
      - Filter out rows lacking Total_C
      - Compute annual mean of all numeric columns
      - Derive seven percentage metrics (using that year’s Total_C)
      - Derive average concentration in sedimentation area (Trapped_SOC_Dam)
        using total grid-cell count as the denominator
    Returns:
      - means: pd.Series of annual means for each numeric column + Trapped_SOC_Dam
      - mets:  pd.Series of the seven percentage metrics (in %)
    """
    base_dir = OUTPUT_DIR / "Data" / "SOC_Future 5" / scenario

    monthly_dfs = []
    dam_ratios = []
    low_ratios = []
    dam_avg_concs = []  # collect average concentration in sedimentation area each month

    for m in range(1, 13):
        fn = f"SOC_terms_{year}_{m:02d}_River.parquet"
        fp = base_dir / fn
        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp}")

        df = pd.read_parquet(fp)
        df = df[df["Total_C"].notna()]
        monthly_dfs.append(df)

        total_c = df["Total_C"].sum()
        if total_c > 0:
            # Dam and low-point percentages
            dam_sum = df.loc[df["Region"] == "sedimentation area", "Total_C"].sum()
            low_sum = df.loc[df["Low point"] == "True", "Total_C"].sum()
            dam_ratios.append(dam_sum / total_c * 100)
            low_ratios.append(low_sum / total_c * 100)

            # Average concentration in sedimentation area per total grid count
            num_total = len(df)
            dam_avg = dam_sum / num_total if num_total > 0 else 0.0
            dam_avg_concs.append(dam_avg)
        else:
            dam_ratios.append(0.0)
            low_ratios.append(0.0)
            dam_avg_concs.append(0.0)

    combined = pd.concat(monthly_dfs, ignore_index=True)
    means = combined.mean(numeric_only=True)

    # ----- compute average concentration in sedimentation area (mean over months) -----
    trapped_dam_avg_conc = sum(dam_avg_concs) / len(dam_avg_concs)
    means["Trapped_SOC_Dam"] = trapped_dam_avg_conc
    # -------------------------------------------------------------------------------

    # Compute the five base % metrics *using this year’s* Total_C
    erosion_pct    = (means["Erosion_fast"] + means["Erosion_slow"])       / means["Total_C"] * 100
    deposition_pct = (means["Deposition_fast"] + means["Deposition_slow"]) / means["Total_C"] * 100
    vegetation_pct = (means["Vegetation_fast"] + means["Vegetation_slow"]) / means["Total_C"] * 100
    reaction_pct   = (means["Reaction_fast"] + means["Reaction_slow"])     / means["Total_C"] * 100
    river_lost_pct = means["Lost_SOC_River"]                               / means["Total_C"] * 100

    dam_avg_pct = sum(dam_ratios) / len(dam_ratios)
    low_avg_pct = sum(low_ratios) / len(low_ratios)

    mets = pd.Series({
        "Erosion_pct":               erosion_pct,
        "Deposition_pct":            deposition_pct,
        "Vegetation_pct":            vegetation_pct,
        "Reaction_pct":              reaction_pct,
        "River_lost_pct":            river_lost_pct,
        "SOC_trapped_dam_pct":       dam_avg_pct,
        "SOC_trapped_low_point_pct": low_avg_pct,
    })

    return means, mets


if __name__ == "__main__":
    year = int(input("Enter year (1950–2100): ").strip())

    # Get this year’s stats + trapped dam concentration added
    means, mets = annual_summary(year)

    # Get previous year’s mean Total_C for denominator
    if year > 1950:
        means_prev, _ = annual_summary(year - 1)
        prev_total_c = means_prev["Total_C"]
    else:
        prev_total_c = means["Total_C"]

    # Override five pct metrics to use previous‐year denominator
    mets["Erosion_pct"]    = (means["Erosion_fast"]    + means["Erosion_slow"])    / prev_total_c * 100
    mets["Reaction_pct"]   = (means["Reaction_fast"]   + means["Reaction_slow"])   / prev_total_c * 100
    mets["Deposition_pct"] = (means["Deposition_fast"] + means["Deposition_slow"]) / prev_total_c * 100
    mets["Vegetation_pct"] = (means["Vegetation_fast"] + means["Vegetation_slow"]) / prev_total_c * 100
    mets["River_lost_pct"] = (means["Lost_SOC_River"])                                / prev_total_c * 100

    # Drop unneeded columns (Trapped_SOC_Dam stays)
    to_drop = [
        "LAT", "LON",
        "E_t_ha_month",
        "C_factor_month", "K_factor_month", "LS_factor_month",
        "P_factor_month", "R_factor_month"
    ]
    means = means.drop(labels=to_drop, errors="ignore")

    # Prepare tidy DataFrames (Trapped_SOC_Dam now appears in df_means)
    df_means = means.reset_index()
    df_means.columns = ["Metric", "Value (g/kg/month)"]
    df_mets  = mets.reset_index()
    df_mets.columns = ["PctMetric", "PctValue (%)"]

    # g/kg → Pg conversion
    area_m2      = 640_000 * 1e6
    volume_m3    = area_m2 * 0.20
    mass_soil_kg = volume_m3 * 1300
    conv_factor  = mass_soil_kg / 1e15

    n       = max(len(df_means), len(df_mets))
    vals    = df_means["Value (g/kg/month)"].reindex(range(n))
    vals_pg = vals * conv_factor

    out = pd.DataFrame({
        "Metric":              df_means["Metric"].reindex(range(n)),
        "Value (g/kg/month)":  vals,
        "Value (Pg/month)":    vals_pg,
        "":                    [""] * n,
        "PctMetric":           df_mets["PctMetric"].reindex(range(n)),
        "PctValue (%)":        df_mets["PctValue (%)"].reindex(range(n)),
    })

    # Category‐contribution logic (unchanged)
    erosion_sum    = means["Erosion_fast"]    + means["Erosion_slow"]
    reaction_sum   = means["Reaction_fast"]   + means["Reaction_slow"]
    deposition_sum = means["Deposition_fast"] + means["Deposition_slow"]
    vegetation_sum = means["Vegetation_fast"] + means["Vegetation_slow"]

    total_decrease = erosion_sum + reaction_sum
    total_increase = deposition_sum + vegetation_sum

    erosion_contrib    = (erosion_sum    / total_decrease * 100) if total_decrease != 0 else 0.0
    reaction_contrib   = (reaction_sum   / total_decrease * 100) if total_decrease != 0 else 0.0
    deposition_contrib = (deposition_sum / total_increase * 100) if total_increase != 0 else 0.0
    vegetation_contrib = (vegetation_sum / total_increase * 100) if total_increase != 0 else 0.0

    contrib_map = {
        "Erosion_pct":    erosion_contrib,
        "Reaction_pct":   reaction_contrib,
        "Deposition_pct": deposition_contrib,
        "Vegetation_pct": vegetation_contrib,
    }
    category_map = {
        "Erosion_pct":    "Erosion",
        "Reaction_pct":   "Reaction",
        "Deposition_pct": "Deposition",
        "Vegetation_pct": "Vegetation",
    }

    out["Category"]         = out["PctMetric"].map(category_map)
    out["Contribution (%)"] = out["PctMetric"].map(contrib_map)

    # Insert a blank column after 'PctValue (%)'
    insert_at = out.columns.get_loc("PctValue (%)") + 1
    out.insert(insert_at, " ", [""] * len(out))

    # Save
    output_fp = OUTPUT_DIR / f"annual_summary_{year}_{SCENARIO}.csv"
    out.to_csv(output_fp, index=False)
    print(f"\nAll results saved to:\n  {output_fp}\n")
