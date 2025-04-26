import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up file paths (assuming PROCESSED_DIR is defined in your globals module)
from globals import *

# ---------------------------------------------------------------------
# 1. Read the theoretical k values from "fitted_k_C_parameters.csv"
# ---------------------------------------------------------------------
theory_file = os.path.join(PROCESSED_DIR, "fitted_k_C_parameters.csv")
df_theory = pd.read_csv(theory_file)

# ---------------------------------------------------------------------
# 2. Precipitation rates and the measured file names
# ---------------------------------------------------------------------
precip_rates = ["60", "90", "120"]

# We will store results (one row per combination of precipitation and region)
results = []

# ---------------------------------------------------------------------
# 3. Loop through each precipitation rate and each Region type
# ---------------------------------------------------------------------
for rate in precip_rates:
    # Build the measured file path for the current precipitation rate.
    measured_file = os.path.join(PROCESSED_DIR, f"k1_k2_{rate}mm_pre_htgy.csv")
    df_measured = pd.read_csv(measured_file)

    # Loop for both regions: "Erosion" and "Sedimentation"
    for region in ["Erosion", "Sedimentation"]:
        # Filter measured data for the current region. Here we do a case-insensitive match.
        df_meas_region = df_measured[df_measured["Region"].str.lower().str.contains(region.lower())]

        # Extract the measured SOC_k2_slow_pool values (convert to numeric and drop NaNs)
        y_meas = pd.to_numeric(df_meas_region["SOC_k1_fast_pool (1/day)"], errors='coerce').dropna().values

        # Check that we have measured data
        if len(y_meas) == 0:
            print(f"No measured data found for {region} at {rate} mm.")
            continue

        # Filter theoretical data for matching precipitation and region.
        # (Assumes theoretical data contains a "Precipitation" column.
        # Adjust the filtering if the type is numeric vs. string.)
        df_theory_region = df_theory[(df_theory["Region"].str.lower() == region.lower()) &
                                     (df_theory["Precipitation"].astype(str) == rate)]

        if df_theory_region.empty:
            print(f"No theoretical data found for {region} at {rate} mm.")
            continue

        # For simplicity, assume one theoretical value per combination.
        # (Adjust if you have more than one row.)
        k_theory = float(df_theory_region["k (1/day)"].iloc[0])

        # Create a predicted array, replicating the theoretical value to match measured data.
        y_pred = np.full_like(y_meas, k_theory, dtype=float)

        # -----------------------------------------------------------------
        # 4. Compute error metrics: RMSE, MAE, and R^2
        # -----------------------------------------------------------------
        rmse = np.sqrt(mean_squared_error(y_meas, y_pred))
        mae = mean_absolute_error(y_meas, y_pred)
        r2 = r2_score(y_meas, y_pred)

        # -----------------------------------------------------------------
        # 5. Compute the 95% confidence interval for the measured mean.
        # -----------------------------------------------------------------
        mean_meas = np.mean(y_meas)
        std_meas = np.std(y_meas, ddof=1)
        sem_meas = std_meas / np.sqrt(len(y_meas))
        conf_int = stats.t.interval(0.95, len(y_meas) - 1, loc=mean_meas, scale=sem_meas)

        # Gather results for reporting or exporting later.
        result = {
            "Precipitation": rate,
            "Region": region,
            "k_theory": k_theory,
            "Mean_measured": mean_meas,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Conf_int_lower": conf_int[0],
            "Conf_int_upper": conf_int[1],
            "N_measured": len(y_meas)
        }
        results.append(result)

        # Print out the results.
        print(f"Precipitation: {rate} mm, Region: {region}")
        print(f"  Theoretical k: {k_theory:.4f} [1/day]")
        print(f"  Mean measured k: {mean_meas:.4f} [1/day] (n = {len(y_meas)})")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  std: {std_meas:.4f}")
        print(f"  95% CI for measured mean: ({conf_int[0]:.4f}, {conf_int[1]:.4f})")
        print("------------------------------------------------------")

# ---------------------------------------------------------------------
# 6. Optionally, combine results into a DataFrame and save to CSV.
# ---------------------------------------------------------------------
df_results = pd.DataFrame(results)
output_file = os.path.join(PROCESSED_DIR, "validation_results_htgy.csv")
df_results.to_csv(output_file, index=False)
print(f"\nValidation results saved to {output_file}")
