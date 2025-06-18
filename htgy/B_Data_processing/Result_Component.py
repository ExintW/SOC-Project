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
      - Derive seven percentage metrics:
          1. Erosion_pct         = (Erosion_fast + Erosion_slow) / Total_C * 100
          2. Deposition_pct      = (Deposition_fast + Deposition_slow) / Total_C * 100
          3. Vegetation_pct      = (Vegetation_fast + Vegetation_slow) / Total_C * 100
          4. Reaction_pct        = (Reaction_fast + Reaction_slow) / Total_C * 100
          5. River_lost_pct      = Lost_SOC_River / Total_C * 100
          6. SOC_trapped_dam_pct = average of monthly [sum(Total_C for Region=="sedimentation area") / sum(Total_C) * 100]
          7. SOC_trapped_low_point_pct = average of monthly [sum(Total_C for Low point==True) / sum(Total_C) * 100]
    Returns:
      - means: pd.Series of annual means for each numeric column
      - mets:  pd.Series of the seven percentage metrics (in %)
    """
    # Base directory for the chosen scenario
    base_dir = OUTPUT_DIR / "Data" / "SOC_Future 3" / scenario

    # We'll collect all monthly DataFrames and monthly dam/low-point ratios
    monthly_dfs = []
    dam_ratios = []
    low_ratios = []

    # Loop over months 1..12
    for m in range(1, 13):
        # Construct filename: SOC_terms_<year>_<MM>_River.parquet
        fn = f"SOC_terms_{year}_{m:02d}_River.parquet"
        fp = base_dir / fn
        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp}")

        # Read the monthly parquet
        df = pd.read_parquet(fp)
        # Keep only rows where Total_C is present (non-NaN)
        df = df[df["Total_C"].notna()]
        monthly_dfs.append(df)

        # Sum Total_C across grid cells for this month
        total_c = df["Total_C"].sum()
        if total_c > 0:
            # Dam-trapped ratio: fraction of SOC in sedimentation-area grids
            dam_sum = df.loc[df["Region"] == "sedimentation area", "Total_C"].sum()
            dam_ratios.append(dam_sum / total_c * 100)

            # Low-point ratio: fraction of SOC in low-point grids
            low_sum = df.loc[df["Low point"] == "True", "Total_C"].sum()
            low_ratios.append(low_sum / total_c * 100)
        else:
            # If no Total_C, treat ratio as zero
            dam_ratios.append(0.0)
            low_ratios.append(0.0)

    # Concatenate all months into one DataFrame
    combined = pd.concat(monthly_dfs, ignore_index=True)
    # Compute yearly means of every numeric column
    means = combined.mean(numeric_only=True)

    # ----- Compute percentage metrics (1-5) from annual means -----
    # Erosion_pct: fraction of mean SOC lost to erosion
    erosion_pct = (
        means["Erosion_fast"] + means["Erosion_slow"]
    ) / means["Total_C"] * 100

    # Deposition_pct: fraction of mean SOC deposited
    deposition_pct = (
        means["Deposition_fast"] + means["Deposition_slow"]
    ) / means["Total_C"] * 100

    # Vegetation_pct: fraction of mean SOC taken up by vegetation
    vegetation_pct = (
        means["Vegetation_fast"] + means["Vegetation_slow"]
    ) / means["Total_C"] * 100

    # Reaction_pct: fraction of mean SOC lost to reaction/mineralization
    reaction_pct = (
        means["Reaction_fast"] + means["Reaction_slow"]
    ) / means["Total_C"] * 100

    # River_lost_pct: fraction of mean SOC lost downriver
    river_lost_pct = means["Lost_SOC_River"] / means["Total_C"] * 100

    # ----- Compute metrics 6-7: average of monthly dam/low-point ratios -----
    dam_avg_pct = sum(dam_ratios) / len(dam_ratios)  # average across 12 months
    low_avg_pct = sum(low_ratios) / len(low_ratios)

    # Bundle the seven percentage metrics
    mets = pd.Series({
        "Erosion_pct": erosion_pct,
        "Deposition_pct": deposition_pct,
        "Vegetation_pct": vegetation_pct,
        "Reaction_pct": reaction_pct,
        "River_lost_pct": river_lost_pct,
        "SOC_trapped_dam_pct": dam_avg_pct,
        "SOC_trapped_low_point_pct": low_avg_pct,
    })

    return means, mets


if __name__ == "__main__":
    # Prompt the user for a year
    year = int(input("Enter year (1950–2100): ").strip())
    means, mets = annual_summary(year)

    # Drop latitude/longitude and per-month factor columns
    to_drop = [
        "LAT", "LON",
        "E_t_ha_month",
        "C_factor_month", "K_factor_month", "LS_factor_month",
        "P_factor_month", "R_factor_month"
    ]
    means = means.drop(labels=to_drop, errors="ignore")

    # Prepare tidy DataFrames
    df_means = means.reset_index()
    df_means.columns = ["Metric", "Value (g/kg/month)"]
    df_mets = mets.reset_index()
    df_mets.columns = ["PctMetric", "PctValue (%)"]

    # ----- g/kg → Pg conversion -----
    # Soil column volume = area × depth
    #   area = 640,000 km² → 640,000 × 10⁶ m²
    #   depth = 0.20 m
    # Bulk density = 1300 kg/m³ → mass of soil in kg
    area_m2 = 640_000 * 1e6
    volume_m3 = area_m2 * 0.20
    mass_soil_kg = volume_m3 * 1300
    # conv_factor: multiply g SOC per kg soil by this to get g SOC total,
    # then divide by 1e15 to convert grams → petagrams
    conv_factor = mass_soil_kg / 1e15

    # Build the final output table with two unit columns
    n = max(len(df_means), len(df_mets))
    vals = df_means["Value (g/kg/month)"].reindex(range(n))
    vals_pg = vals * conv_factor

    out = pd.DataFrame({
        "Metric":              df_means["Metric"].reindex(range(n)),
        "Value (g/kg/month)":  vals,
        "Value (Pg)":          vals_pg,
        "":                    [""] * n,
        "PctMetric":           df_mets["PctMetric"].reindex(range(n)),
        "PctValue (%)":        df_mets["PctValue (%)"].reindex(range(n)),
    })

    # Save to a single CSV (no extra index column)
    output_fp = OUTPUT_DIR / f"annual_summary_{year}_{SCENARIO}.csv"
    out.to_csv(output_fp, index=False)

    print(f"\nAll results saved to:\n  {output_fp}\n")
