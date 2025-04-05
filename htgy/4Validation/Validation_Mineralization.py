import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # DATA_DIR should be defined here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------------------------------
# 1. Define the functions
#    - The derivative: dy/dt = -k * C * exp(-k t)
#    - The underlying function: y = C * exp(-k t)
# ----------------------------------------------------
def exp_decay_deriv(t, C, k):
    return -k * C * np.exp(-k * t)

def exp_decay(t, C, k):
    return C * np.exp(-k * t)

# ----------------------------------------------------
# 2. Load the CSV file.
# ----------------------------------------------------
csv_path = os.path.join(DATA_DIR, "SOC_Mineralization_Data.csv")
df = pd.read_csv(csv_path)

# ----------------------------------------------------
# 3. Define the time points (days).
#    There are 5 measurements at days: 1, 7, 14, 28, 56.
# ----------------------------------------------------
days = np.array([1, 7, 14, 28, 56], dtype=float)

# ----------------------------------------------------
# 4. Extract the data for each precipitation intensity.
#    For each column, we extract:
#       - 60 precipitation: rows 1 to 6  (Python indices 1:6)
#       - 90 precipitation: rows 6 to 11 (Python indices 6:11)
#       - 120 precipitation: rows 11 to 16 (Python indices 11:16)
# ----------------------------------------------------
erosion_60  = -pd.to_numeric(df["Erosion Area"].iloc[1:6], errors='coerce').values
erosion_90  = -pd.to_numeric(df["Erosion Area"].iloc[6:11], errors='coerce').values
erosion_120 = -pd.to_numeric(df["Erosion Area"].iloc[11:16], errors='coerce').values
print("Erosion Data:")
print(erosion_60, erosion_90, erosion_120)

sed_60  = -pd.to_numeric(df["Sedimentation Area"].iloc[1:6], errors='coerce').values
sed_90  = -pd.to_numeric(df["Sedimentation Area"].iloc[6:11], errors='coerce').values
sed_120 = -pd.to_numeric(df["Sedimentation Area"].iloc[11:16], errors='coerce').values
print("Sedimentation Data:")
print(sed_60, sed_90, sed_120)

# ----------------------------------------------------
# 5. Define a helper to fit the derivative data.
#    The initial guess uses:
#       C_guess = |y0| / (k * exp(-k*t0))   with k_guess=0.1.
# ----------------------------------------------------
def fit_derivative(x, y):
    k_guess = 0.1
    C_guess = abs(y[0]) / (k_guess * np.exp(-k_guess * x[0]))
    p0 = [C_guess, k_guess]
    popt, _ = curve_fit(exp_decay_deriv, x, y, p0=p0, maxfev=10000)
    return popt

# ----------------------------------------------------
# 6. Fit the derivative model for each dataset.
#    (Using the same time vector for each dataset.)
# ----------------------------------------------------
# Erosion fits
C_e60, k_e60   = fit_derivative(days, erosion_60)
C_e90, k_e90   = fit_derivative(days, erosion_90)
C_e120, k_e120 = fit_derivative(days, erosion_120)

# Sedimentation fits
C_s60, k_s60   = fit_derivative(days, sed_60)
C_s90, k_s90   = fit_derivative(days, sed_90)
C_s120, k_s120 = fit_derivative(days, sed_120)

# ----------------------------------------------------
# 7. Print the fitted underlying exponential functions.
# ----------------------------------------------------
print("Fitted Erosion Functions (y = C * exp(-k t)):")
print(f"  60 precipitation: y = {C_e60:.4f} * exp(-{k_e60:.4f} t)")
print(f"  90 precipitation: y = {C_e90:.4f} * exp(-{k_e90:.4f} t)")
print(f"  120 precipitation: y = {C_e120:.4f} * exp(-{k_e120:.4f} t)")

print("\nFitted Sedimentation Functions (y = C * exp(-k t)):")
print(f"  60 precipitation: y = {C_s60:.4f} * exp(-{k_s60:.4f} t)")
print(f"  90 precipitation: y = {C_s90:.4f} * exp(-{k_s90:.4f} t)")
print(f"  120 precipitation: y = {C_s120:.4f} * exp(-{k_s120:.4f} t)")

# ----------------------------------------------------
# 8. Create a fine time vector for smooth plotting.
# ----------------------------------------------------
time_fine = np.linspace(days.min(), days.max(), 200)

# ----------------------------------------------------
# 9. Visualization: Four Figures in Total
# ----------------------------------------------------
# Figure 1: Erosion Area derivative (dy/dt) with fitted curve
plt.figure(figsize=(8, 5))
plt.scatter(days, erosion_60, color="blue", label="60 mm/h data")
plt.plot(time_fine, exp_decay_deriv(time_fine, C_e60, k_e60),
         color="blue", linestyle="--",
         label=f"60 mm/h fit: y = {C_e60:.2f} exp(-{k_e60:.2f} t)")

plt.scatter(days, erosion_90, color="red", label="90 mm/h data")
plt.plot(time_fine, exp_decay_deriv(time_fine, C_e90, k_e90),
         color="red", linestyle="--",
         label=f"90 mm/h fit: y = {C_e90:.2f} exp(-{k_e90:.2f} t)")

plt.scatter(days, erosion_120, color="green", label="120 mm/h data")
plt.plot(time_fine, exp_decay_deriv(time_fine, C_e120, k_e120),
         color="green", linestyle="--",
         label=f"120 mm/h fit: y = {C_e120:.2f} exp(-{k_e120:.2f} t)")

plt.title("Erosion Area: dy/dt Fitted to -k C exp(-k t)")
plt.xlabel("Days")
plt.ylabel("dy/dt")
plt.legend()
plt.grid(True)
plt.show()

# Figure 2: Sedimentation Area derivative (dy/dt) with fitted curve
plt.figure(figsize=(8, 5))
plt.scatter(days, sed_60, color="blue", label="60 mm/h data")
plt.plot(time_fine, exp_decay_deriv(time_fine, C_s60, k_s60),
         color="blue", linestyle="--",
         label=f"60 mm/h fit: y = {C_s60:.2f} exp(-{k_s60:.2f} t)")

plt.scatter(days, sed_90, color="red", label="90 mm/h data")
plt.plot(time_fine, exp_decay_deriv(time_fine, C_s90, k_s90),
         color="red", linestyle="--",
         label=f"90 mm/h fit: y = {C_s90:.2f} exp(-{k_s90:.2f} t)")

plt.scatter(days, sed_120, color="green", label="120 mm/h data")
plt.plot(time_fine, exp_decay_deriv(time_fine, C_s120, k_s120),
         color="green", linestyle="--",
         label=f"120 mm/h fit: y = {C_s120:.2f} exp(-{k_s120:.2f} t)")

plt.title("Sedimentation Area: dy/dt Fitted to -k C exp(-k t)")
plt.xlabel("Days")
plt.ylabel("dy/dt")
plt.legend()
plt.grid(True)
plt.show()

# Figure 3: Underlying fitted function y = C exp(-k t) for Erosion Area
plt.figure(figsize=(8, 5))
plt.plot(time_fine, exp_decay(time_fine, C_e60, k_e60),
         color="blue", linestyle="--",
         label=f"60 mm/h: y = {C_e60:.2f} exp(-{k_e60:.2f} t)")
plt.plot(time_fine, exp_decay(time_fine, C_e90, k_e90),
         color="red", linestyle="--",
         label=f"90 mm/h: y = {C_e90:.2f} exp(-{k_e90:.2f} t)")
plt.plot(time_fine, exp_decay(time_fine, C_e120, k_e120),
         color="green", linestyle="--",
         label=f"120 mm/h: y = {C_e120:.2f} exp(-{k_e120:.2f} t)")

plt.title("Erosion Area: Fitted Function y = C exp(-k t)")
plt.xlabel("Days")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Figure 4: Underlying fitted function y = C exp(-k t) for Sedimentation Area
plt.figure(figsize=(8, 5))
plt.plot(time_fine, exp_decay(time_fine, C_s60, k_s60),
         color="blue", linestyle="--",
         label=f"60 mm/h: y = {C_s60:.2f} exp(-{k_s60:.2f} t)")
plt.plot(time_fine, exp_decay(time_fine, C_s90, k_s90),
         color="red", linestyle="--",
         label=f"90 mm/h: y = {C_s90:.2f} exp(-{k_s90:.2f} t)")
plt.plot(time_fine, exp_decay(time_fine, C_s120, k_s120),
         color="green", linestyle="--",
         label=f"120 mm/h: y = {C_s120:.2f} exp(-{k_s120:.2f} t)")

plt.title("Sedimentation Area: Fitted Function y = C exp(-k t)")
plt.xlabel("Days")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
