import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # DATA_DIR should be defined here

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------------------------------
# 1. Define the total mineralization function.
#    The model is: y = C * (1 - exp(-k*t))
# ----------------------------------------------------
def total_mineralization(t, C, k):
    return C * (1 - np.exp(-k * t))

# ----------------------------------------------------
# 2. Load the CSV file.
# ----------------------------------------------------
csv_path = os.path.join(DATA_DIR, "SOC_Total_Mineralization_Data.csv")
df = pd.read_csv(csv_path)

# ----------------------------------------------------
# 3. Define the measurement time points (days).
#    Measurements are taken on days: 1, 7, 14, 28, 56.
# ----------------------------------------------------
days = np.array([1, 7, 14, 28, 56], dtype=float)

# ----------------------------------------------------
# 4. Extract Total Mineralization data for Erosion and Sedimentation.
#
#    It is assumed that the CSV file has columns:
#      - "Erosion Area"
#      - "Sedimentation Area"
#
#    For each precipitation intensity the rows are:
#       - 60 mm/h: rows 1 to 6   (Python indices 1:6)
#       - 90 mm/h: rows 6 to 11  (Python indices 6:11)
#       - 120 mm/h: rows 11 to 16 (Python indices 11:16)
# ----------------------------------------------------
# Erosion Total Mineralization Data
erosion_total_60  = pd.to_numeric(df["Erosion Area"].iloc[1:6], errors='coerce').values
erosion_total_90  = pd.to_numeric(df["Erosion Area"].iloc[6:11], errors='coerce').values
erosion_total_120 = pd.to_numeric(df["Erosion Area"].iloc[11:16], errors='coerce').values

# Sedimentation Total Mineralization Data
sed_total_60  = pd.to_numeric(df["Sedimentation Area"].iloc[1:6], errors='coerce').values
sed_total_90  = pd.to_numeric(df["Sedimentation Area"].iloc[6:11], errors='coerce').values
sed_total_120 = pd.to_numeric(df["Sedimentation Area"].iloc[11:16], errors='coerce').values

print("Erosion Total Mineralization Data:")
print("60 mm/h:", erosion_total_60)
print("90 mm/h:", erosion_total_90)
print("120 mm/h:", erosion_total_120)

print("\nSedimentation Total Mineralization Data:")
print("60 mm/h:", sed_total_60)
print("90 mm/h:", sed_total_90)
print("120 mm/h:", sed_total_120)

# ----------------------------------------------------
# 5. Define a helper function for fitting total mineralization data.
#
#    We use an initial guess:
#      - k_guess = 0.1
#      - C_guess = last measured value (assumed near saturation)
# ----------------------------------------------------
def fit_total_mineralization(x, y):
    k_guess = 0.1
    C_guess = y[-1]
    p0 = [C_guess, k_guess]
    popt, _ = curve_fit(total_mineralization, x, y, p0=p0, maxfev=10000)
    return popt

# ----------------------------------------------------
# 6. Fit the total mineralization model for each dataset.
# ----------------------------------------------------
# Erosion fits
C_e60, k_e60   = fit_total_mineralization(days, erosion_total_60)
C_e90, k_e90   = fit_total_mineralization(days, erosion_total_90)
C_e120, k_e120 = fit_total_mineralization(days, erosion_total_120)

# Sedimentation fits
C_s60, k_s60   = fit_total_mineralization(days, sed_total_60)
C_s90, k_s90   = fit_total_mineralization(days, sed_total_90)
C_s120, k_s120 = fit_total_mineralization(days, sed_total_120)

# ----------------------------------------------------
# 7. Print the fitted functions.
# ----------------------------------------------------
print("\nFitted Total Mineralization Functions (y = C * (1 - exp(-k*t))):")
print("Erosion:")
print(f"  60 mm/h: y = {C_e60:.4f} * (1 - exp(-{k_e60:.4f} t))")
print(f"  90 mm/h: y = {C_e90:.4f} * (1 - exp(-{k_e90:.4f} t))")
print(f"  120 mm/h: y = {C_e120:.4f} * (1 - exp(-{k_e120:.4f} t))")

print("\nSedimentation:")
print(f"  60 mm/h: y = {C_s60:.4f} * (1 - exp(-{k_s60:.4f} t))")
print(f"  90 mm/h: y = {C_s90:.4f} * (1 - exp(-{k_s90:.4f} t))")
print(f"  120 mm/h: y = {C_s120:.4f} * (1 - exp(-{k_s120:.4f} t))")

# ----------------------------------------------------
# 7.5 Save the fitted parameters into an output CSV file.
# ----------------------------------------------------
# Create a results dictionary combining data for Erosion and Sedimentation
results = {
    "Region": ["Erosion", "Erosion", "Erosion", "Sedimentation", "Sedimentation", "Sedimentation"],
    "Precipitation": [60, 90, 120, 60, 90, 120],
    "C": [C_e60, C_e90, C_e120, C_s60, C_s90, C_s120],
    "k (1/day)": [k_e60, k_e90, k_e120, k_s60, k_s90, k_s120],
}

results_df = pd.DataFrame(results)
output_csv_path = os.path.join(PROCESSED_DIR, "fitted_k_C_parameters.csv")
results_df.to_csv(output_csv_path, index=False)
print(f"\nFitted parameters saved to: {output_csv_path}")

# ----------------------------------------------------
# 8. Create a fine time vector for smooth plotting.
# ----------------------------------------------------
time_fine = np.linspace(days.min(), days.max(), 200)

# ----------------------------------------------------
# 9. Visualization
#
#    We create two figures: one for the erosion area and one for the sedimentation area.
# ----------------------------------------------------

# Figure for Erosion Total Mineralization
plt.figure(figsize=(8, 5))
plt.scatter(days, erosion_total_60, color="blue", label="60 mm/h data")
plt.plot(time_fine, total_mineralization(time_fine, C_e60, k_e60),
         color="blue", linestyle="--",
         label=f"60 mm/h fit: y = {C_e60:.4f}(1-exp(-{k_e60:.4f}t))")

plt.scatter(days, erosion_total_90, color="red", label="90 mm/h data")
plt.plot(time_fine, total_mineralization(time_fine, C_e90, k_e90),
         color="red", linestyle="--",
         label=f"90 mm/h fit: y = {C_e90:.4f}(1-exp(-{k_e90:.4f}t))")

plt.scatter(days, erosion_total_120, color="green", label="120 mm/h data")
plt.plot(time_fine, total_mineralization(time_fine, C_e120, k_e120),
         color="green", linestyle="--",
         label=f"120 mm/h fit: y = {C_e120:.4f}(1-exp(-{k_e120:.4f}t))")

plt.title("Erosion Total Mineralization: y = C * (1 - exp(-k t))")
plt.xlabel("Days")
plt.ylabel("Erosion Total Mineralization")
plt.legend()
plt.grid(True)
plt.show()

# Figure for Sedimentation Total Mineralization
plt.figure(figsize=(8, 5))
plt.scatter(days, sed_total_60, color="blue", label="60 mm/h data")
plt.plot(time_fine, total_mineralization(time_fine, C_s60, k_s60),
         color="blue", linestyle="--",
         label=f"60 mm/h fit: y = {C_s60:.4f}(1-exp(-{k_s60:.4f}t))")

plt.scatter(days, sed_total_90, color="red", label="90 mm/h data")
plt.plot(time_fine, total_mineralization(time_fine, C_s90, k_s90),
         color="red", linestyle="--",
         label=f"90 mm/h fit: y = {C_s90:.4f}(1-exp(-{k_s90:.4f}t))")

plt.scatter(days, sed_total_120, color="green", label="120 mm/h data")
plt.plot(time_fine, total_mineralization(time_fine, C_s120, k_s120),
         color="green", linestyle="--",
         label=f"120 mm/h fit: y = {C_s120:.4f}(1-exp(-{k_s120:.4f}t))")

plt.title("Sedimentation Total Mineralization: y = C * (1 - exp(-k t))")
plt.xlabel("Days")
plt.ylabel("Sedimentation Total Mineralization")
plt.legend()
plt.grid(True)
plt.show()

