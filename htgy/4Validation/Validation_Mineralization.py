import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # DATA_DIR should be defined here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp_decay_deriv(t, C, k):
    return -k * exp_decay(t, C, k)

def exp_decay(t, C, k):
    return C * np.exp(-k * t)


def model_deriv(t, k, y_init):
    """
    计算在时间 t 下的导数 dy/dt = -k · y(t),
    其中 y(t) = y_init * exp(-k t),
    y_init 为给定的初始值(常量), k 为待拟合参数.
    """
    # y(t) = y_init * exp(-k t)
    y_t = y_init * np.exp(-k * t)
    return -k * y_t  # 即 -k·y

def fit_k_for_deriv(x, dCdt_obs, y_init):
    """
    用曲线拟合的方式, 根据观测的导数 dCdt_obs, 
    拟合 dy/dt = -k * y(t), y(0)=y_init, 仅仅求出 k.
    """
    def func_to_fit(time, k):
        return model_deriv(time, k, y_init)
    
    # k 的初值猜测
    k_guess = 0.1
    popt, _ = curve_fit(func_to_fit, x, dCdt_obs, p0=[k_guess], maxfev=10000)
    return popt[0]  # 只返回最优k

csv_path = os.path.join(DATA_DIR, "SOC_Mineralization_Data.csv")
df = pd.read_csv(csv_path)

days = np.array([1, 7, 14, 28, 56], dtype=float)

# ----------------------------------------------------
# Extract the data for each precipitation intensity.
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

C_erosion = 3.56 * 1000  # 3560 mg/kg
C_sed     = 3.72 * 1000  # 3720 mg/kg

# ----------------------------------------------------
# Fit the derivative model for each dataset.
# (Using the same time vector for each dataset.)
# ----------------------------------------------------
# Erosion fits
k_e60   = fit_k_for_deriv(days, erosion_60, C_erosion)
k_e90   = fit_k_for_deriv(days, erosion_90, C_erosion)
k_e120 = fit_k_for_deriv(days, erosion_120, C_erosion)

# Sedimentation fits
k_s60   = fit_k_for_deriv(days, sed_60, C_sed)
k_s90   = fit_k_for_deriv(days, sed_90, C_sed)
k_s120 = fit_k_for_deriv(days, sed_120, C_sed)

# ----------------------------------------------------
# Print the fitted underlying exponential functions.
# ----------------------------------------------------
print("\nFitted k values (dy/dt = -k·y):")
print(f"Erosion 60 mm/h: k={k_e60}, y0={C_erosion:.1f} mg/kg")
print(f"Erosion 90 mm/h: k={k_e90}, y0={C_erosion:.1f} mg/kg")
print(f"Erosion120 mm/h: k={k_e120}, y0={C_erosion:.1f} mg/kg")

print(f"Sed     60 mm/h: k={k_s60}, y0={C_sed:.1f} mg/kg")
print(f"Sed     90 mm/h: k={k_s90}, y0={C_sed:.1f} mg/kg")
print(f"Sed    120 mm/h: k={k_s120}, y0={C_sed:.1f} mg/kg")

# ----------------------------------------------------
# Create a fine time vector for smooth plotting.
# ----------------------------------------------------
time_fine = np.linspace(days.min(), days.max(), 200)

# ----------------------------------------------------
# Visualization: Four Figures in Total
# ----------------------------------------------------
# Figure 1: Erosion Area derivative (dy/dt) with fitted curve
plt.figure(figsize=(8, 5))
plt.scatter(days, erosion_60, color="blue", label="60 mm/h data") 
plt.plot(time_fine, exp_decay_deriv(time_fine, C_erosion, k_e60),
         color="blue", linestyle="--",
         label=f"60 mm/h fit: y = {C_erosion:.2f} exp(-{k_e60:.2f} t)")

plt.scatter(days, erosion_90, color="red", label="90 mm/h data")
plt.plot(time_fine, exp_decay_deriv(time_fine, C_erosion, k_e90),
         color="red", linestyle="--",
         label=f"90 mm/h fit: y = {C_erosion:.2f} exp(-{k_e90:.2f} t)")

plt.scatter(days, erosion_120, color="green", label="120 mm/h data")
plt.plot(time_fine, exp_decay_deriv(time_fine, C_erosion, k_e120),
         color="green", linestyle="--",
         label=f"120 mm/h fit: y = {C_erosion:.2f} exp(-{k_e120:.2f} t)")

plt.title("Erosion Area: dy/dt Fitted to -k C exp(-k t)")
plt.xlabel("Days")
plt.ylabel("dy/dt")
plt.legend()
plt.grid(True)
plt.show()

# Figure 2: Sedimentation Area derivative (dy/dt) with fitted curve
plt.figure(figsize=(8, 5))
plt.scatter(days, sed_60, color="blue", label="60 mm/h data")
plt.plot(time_fine, exp_decay_deriv(time_fine, C_sed, k_s60),
         color="blue", linestyle="--",
         label=f"60 mm/h fit: y = {C_sed:.2f} exp(-{k_s60:.2f} t)")

plt.scatter(days, sed_90, color="red", label="90 mm/h data")
plt.plot(time_fine, exp_decay_deriv(time_fine, C_sed, k_s90),
         color="red", linestyle="--",
         label=f"90 mm/h fit: y = {C_sed:.2f} exp(-{k_s90:.2f} t)")

plt.scatter(days, sed_120, color="green", label="120 mm/h data")
plt.plot(time_fine, exp_decay_deriv(time_fine, C_sed, k_s120),
         color="green", linestyle="--",
         label=f"120 mm/h fit: y = {C_sed:.2f} exp(-{k_s120:.2f} t)")

plt.title("Sedimentation Area: dy/dt Fitted to -k C exp(-k t)")
plt.xlabel("Days")
plt.ylabel("dy/dt")
plt.legend()
plt.grid(True)
plt.show()

# Figure 3: Underlying fitted function y = C exp(-k t) for Erosion Area
plt.figure(figsize=(8, 5))
plt.plot(time_fine, exp_decay(time_fine, C_erosion, k_e60),
         color="blue", linestyle="--",
         label=f"60 mm/h: y = {C_erosion:.2f} exp(-{k_e60:.2f} t)")
plt.plot(time_fine, exp_decay(time_fine, C_erosion, k_e90),
         color="red", linestyle="--",
         label=f"90 mm/h: y = {C_erosion:.2f} exp(-{k_e90:.2f} t)")
plt.plot(time_fine, exp_decay(time_fine, C_erosion, k_e120),
         color="green", linestyle="--",
         label=f"120 mm/h: y = {C_erosion:.2f} exp(-{k_e120:.2f} t)")

plt.title("Erosion Area: Fitted Function y = C exp(-k t)")
plt.xlabel("Days")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Figure 4: Underlying fitted function y = C exp(-k t) for Sedimentation Area
plt.figure(figsize=(8, 5))
plt.plot(time_fine, exp_decay(time_fine, C_sed, k_s60),
         color="blue", linestyle="--",
         label=f"60 mm/h: y = {C_sed:.2f} exp(-{k_s60:.2f} t)")
plt.plot(time_fine, exp_decay(time_fine, C_sed, k_s90),
         color="red", linestyle="--",
         label=f"90 mm/h: y = {C_sed:.2f} exp(-{k_s90:.2f} t)")
plt.plot(time_fine, exp_decay(time_fine, C_sed, k_s120),
         color="green", linestyle="--",
         label=f"120 mm/h: y = {C_sed:.2f} exp(-{k_s120:.2f} t)")

plt.title("Sedimentation Area: Fitted Function y = C exp(-k t)")
plt.xlabel("Days")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
