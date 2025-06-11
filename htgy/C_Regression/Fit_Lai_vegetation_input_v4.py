import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import re
import sys
import os
import geopandas as gpd
from shapely.geometry import Point

# Append parent directory to system path for dependency resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from htgy.globals import *

# --------------------------------------------------
#                Data Reading and Cleaning
# --------------------------------------------------

# Use the processed file from the previous code as the input file
csv_file = PROCESSED_DIR / "Vege_Input_Data_CMIP6_Outlier_Removed.csv"
df = pd.read_csv(csv_file, encoding='ISO-8859-1')

# Standardize column names by stripping whitespace
df.columns = df.columns.str.strip()

# Define the relevant column names
lai_col = "2007 LAI"
soc_col = "SOC Monthly Increase (g/kg/month)"

# Cleaning function: remove non-numeric characters
def clean_numeric(value):
    if isinstance(value, str):
        v = re.sub(r"[^\d\.-]", "", value)
        try:
            return float(v) if v else np.nan
        except ValueError:
            return np.nan
    return value

# Apply cleaning
df[lai_col] = df[lai_col].apply(clean_numeric)
df[soc_col] = df[soc_col].apply(clean_numeric)

# Drop rows with NaNs in those two columns
df_cleaned = df.dropna(subset=[lai_col, soc_col])

# Initial outlier removal by SOC range (e.g. keep 0–100 g/kg/month)
df_cleaned = df_cleaned[(df_cleaned[soc_col] >= -100) & (df_cleaned[soc_col] <= 100)]

# --------------------------------------------------
#      Manual Boundary‐Line Outlier Removal
# --------------------------------------------------
# Define your two lines: y <= m_upper*x + b_upper and y >= m_lower*x + b_lower
# Replace these slope/intercept values with the ones you drew:
m_upper, b_upper =  0.13, 0.13333   # upper boundary line
m_lower, b_lower = 0.13,  -0.06666   # lower boundary line

# Build mask of points BETWEEN those two lines
x_vals = df_cleaned[lai_col].to_numpy()
y_vals = df_cleaned[soc_col].to_numpy()
mask_between = (
    (y_vals <=  m_upper * x_vals + b_upper) &
    (y_vals >=  m_lower * x_vals + b_lower)
)

# Apply the mask
df_cleaned = df_cleaned.loc[mask_between].copy()

# Report how many you dropped
num_total = len(x_vals)
num_kept  = mask_between.sum()
print(f"Dropped {num_total - num_kept} points outside your boundary lines; keeping {num_kept} points.")

# --------------------------------------------------
#       Recompute Positive/Negative Subsets
# --------------------------------------------------
negative_points = df_cleaned[df_cleaned[soc_col] < 0]
positive_points = df_cleaned[df_cleaned[soc_col] > 0]
print(f"Total data points: {len(df_cleaned)}")
print(f"Positive SOC points: {len(positive_points)}")
print(f"Zero or negative SOC points: {len(negative_points)}")

# Extract data for regression
X = df_cleaned[lai_col].values
Y = df_cleaned[soc_col].values

# --------------------------------------------------
#              Define Regression Models
# --------------------------------------------------
def linear_model(x, a, b):
    return a * x + b

def logarithmic_model(x, a, b):
    return a * np.log(x) + b

def polynomial2_model(x, a, b, c):
    return a + b*x + c*x**2

def polynomial3_model(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3

def polynomial4_model(x, a, b, c, d, e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4

def polynomial5_model(x, a, b, c, d, e, f):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5

models = {
    "Linear": linear_model,
    "Logarithmic": logarithmic_model,
    "Polynomial_2": polynomial2_model,
    "Polynomial_3": polynomial3_model,
    "Polynomial_4": polynomial4_model,
    "Polynomial_5": polynomial5_model,
}

# --------------------------------------------------
#            Error Analysis: MSE, RMSE, R²
# --------------------------------------------------
def compute_errors(x, y, func, params):
    y_pred = func(x, *params)
    resid  = y - y_pred
    sse    = np.sum(resid**2)
    mse    = sse / len(x)
    rmse   = np.sqrt(mse)
    tss    = np.sum((y - np.mean(y))**2)
    r2     = 1 - sse/tss if tss != 0 else np.nan
    return mse, rmse, r2

# --------------------------------------------------
#     Fit Models and Print Equations / Metrics
# --------------------------------------------------
fit_results   = {}
error_results = {}

for name, func in models.items():
    try:
        params, _ = curve_fit(func, X, Y, maxfev=5000)
        fit_results[name] = params
        mse, rmse, r2 = compute_errors(X, Y, func, params)
        error_results[name] = (mse, rmse, r2)

        # build equation string
        if name == "Linear":
            eq = f"y = {params[0]:.4f} x + {params[1]:.4f}"
        elif name == "Logarithmic":
            eq = f"y = {params[0]:.4f} ln(x) + {params[1]:.4f}"
        else:
            coeffs = " + ".join(f"{p:.4f} x^{i}" if i>0 else f"{p:.4f}"
                                for i,p in enumerate(params))
            eq = f"y = {coeffs}"

        print(f"{name} Model:")
        print(f"  Equation: {eq}")
        print(f"  MSE: {mse:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")
        print("-"*50)

    except Exception as e:
        print(f"Could not fit {name}: {e}")
        print("-"*50)

# --------------------------------------------------
#                 Plot and Visualization
# --------------------------------------------------
x_plot = np.linspace(X.min(), X.max(), 200)

plt.figure(figsize=(12, 6))

# dropped (for reference) in grey
# Note: we can reconstruct dropped by inverting mask_between if desired

# kept points
sns.scatterplot(x=positive_points[lai_col], y=positive_points[soc_col],
                color="blue", alpha=0.7, label="Positive SOC")
sns.scatterplot(x=negative_points[lai_col], y=negative_points[soc_col],
                color="red",  alpha=0.7, label="Negative SOC")

# boundary lines themselves
y_up   = m_upper * x_plot + b_upper
y_low  = m_lower * x_plot + b_lower
plt.plot(x_plot, y_up,  color="black", linestyle="--", label="Upper boundary")
plt.plot(x_plot, y_low, color="black", linestyle="--", label="Lower boundary")

# fitted curves
for name, func in models.items():
    if name in fit_results:
        y_fit = func(x_plot, *fit_results[name])
        plt.plot(x_plot, y_fit, linewidth=2, label=f"{name} Fit")

plt.xlabel("2007 LAI")
plt.ylabel("SOC Monthly Increase (g/kg/month)")
plt.title("Fits after Manual Outlier Removal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
