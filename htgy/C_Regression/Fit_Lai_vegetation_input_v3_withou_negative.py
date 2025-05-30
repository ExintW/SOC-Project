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
csv_file = PROCESSED_DIR / "Vege_Input_Data_Updated.csv"
df = pd.read_csv(csv_file, encoding='ISO-8859-1')

# Standardize column names by stripping whitespace
df.columns = df.columns.str.strip()

# Define the relevant column names (using processed file columns)
lai_col = "2007 LAI"
soc_col = "SOC Monthly Increase (g/kg/month)"

# Print the original SOC column values (first 10 rows) for inspection
print("Original SOC values before cleaning:\n", df[soc_col].head(10))


# Cleaning function: remove special characters and keep only digits, dot, and minus sign
def clean_numeric(value):
    if isinstance(value, str):
        value = re.sub(r'[^\d\.-]', '', value)  # keep only digits, dot, and minus sign
        try:
            return float(value) if value else np.nan
        except ValueError:
            return np.nan
    return value


# Apply the cleaning function to both the SOC and LAI columns
df[soc_col] = df[soc_col].apply(clean_numeric)
df[lai_col] = df[lai_col].apply(clean_numeric)

# Print the cleaned SOC values (first 10 rows)
print("Cleaned SOC values:\n", df[soc_col].head(10))

# Drop rows with NaN values in the SOC or LAI columns
df_cleaned = df.dropna(subset=[lai_col, soc_col])

# Remove outliers: rows where the SOC monthly increase > 1.5
df_cleaned = df_cleaned[df_cleaned[soc_col] <= 1.5]

# Remove negative SOC monthly increase points
df_cleaned = df_cleaned[df_cleaned[soc_col] >= 0]

# Count the number of positive SOC data points after filtering
print(f"Total data points (after outlier and negative removal): {len(df_cleaned)}")

# For visualization, assign positive_points (all data now is positive)
positive_points = df_cleaned

# Extract data for regression: X (2007 LAI) and Y (Monthly SOC Increase)
X = df_cleaned[lai_col].values
Y = df_cleaned[soc_col].values


# --------------------------------------------------
#              Define Regression Models
# --------------------------------------------------

# 1. Linear model: y = a * x + b
def linear_model(x, a, b):
    return a * x + b


# 2. Exponential model: y = a * exp(b * x)
def exponential_model(x, a, b):
    return a * np.exp(b * x)


# 3. Power-law model: y = a * x^b
def power_model(x, a, b):
    return a * np.power(x, b)


# 4. Logarithmic model: y = a * log(x) + b
def logarithmic_model(x, a, b):
    return a * np.log(x) + b


# 5. Quadratic polynomial model: y = a + b*x + c*x^2
def polynomial2_model(x, a, b, c):
    return a + b * x + c * (x ** 2)


# --------------------------------------------------
#            Error Analysis: MSE, RMSE, R²
# --------------------------------------------------
def compute_errors(x, y, model_func, params):
    """
    Compute error metrics given x, y, a model function, and fitted parameters.
    Returns:
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
        r2: Coefficient of Determination (R²)
    """
    y_pred = model_func(x, *params)
    residuals = y - y_pred
    sse = np.sum(residuals ** 2)  # Sum of Squared Errors
    mse = sse / len(x)
    rmse = np.sqrt(mse)
    tss = np.sum((y - np.mean(y)) ** 2)  # Total Sum of Squares
    r2 = 1 - sse / tss if tss != 0 else 0
    return mse, rmse, r2


# --------------------------------------------------
#             Fit Models and Print Equation / Error Results
# --------------------------------------------------
models = {
    "Linear": linear_model,
    "Exponential": exponential_model,
    "Power-Law": power_model,
    "Logarithmic": logarithmic_model,
    "Polynomial_2": polynomial2_model
}

fit_results = {}  # Dictionary to store fitted parameters for each model
error_results = {}  # Dictionary to store error metrics for each model

for name, model_func in models.items():
    try:
        # Fit the model using curve_fit (note: some models require x > 0, e.g., logarithmic)
        params, _ = curve_fit(model_func, X, Y, maxfev=5000)
        fit_results[name] = params
        mse, rmse, r2 = compute_errors(X, Y, model_func, params)
        error_results[name] = (mse, rmse, r2)

        # Create an equation string based on the model type
        if name == "Linear":
            eq_str = f"y = {params[0]:.8f} * x + {params[1]:.8f}"
        elif name == "Exponential":
            eq_str = f"y = {params[0]:.8f} * exp({params[1]:.8f} * x)"
        elif name == "Power-Law":
            eq_str = f"y = {params[0]:.8f} * x^{params[1]:.8f}"
        elif name == "Logarithmic":
            eq_str = f"y = {params[0]:.8f} * ln(x) + {params[1]:.8f}"
        elif name == "Polynomial_2":
            eq_str = f"y = {params[0]:.8f} + {params[1]:.8f} * x + {params[2]:.8f} * x^2"
        else:
            eq_str = "Equation not defined"

        print(f"{name} Model:")
        print(f"  Equation: {eq_str}")
        print(f"  MSE  = {mse:.8f}")
        print(f"  RMSE = {rmse:.8f}")
        print(f"  R²   = {r2:.8f}")
        print("-" * 50)
    except Exception as e:
        print(f"Could not fit {name} model: {e}")
        print("-" * 50)

# --------------------------------------------------
#                 Plot and Visualization
# --------------------------------------------------

# Generate equally spaced x-values for plotting fitted curves
x_vals = np.linspace(X.min(), X.max(), 100)

plt.figure(figsize=(10, 6))
# Plot positive SOC points as scatter plots (all data is now positive)
sns.scatterplot(x=positive_points[lai_col], y=positive_points[soc_col],
                color="blue", alpha=0.7, label="Positive SOC")

# Plot the fitted curves for each model with equation annotations
for name, model_func in models.items():
    if name in fit_results:
        params = fit_results[name]
        y_fit = model_func(x_vals, *params)
        plt.plot(x_vals, y_fit, label=f"{name} Fit")
        if name == "Linear":
            eq_text = f"y={params[0]:.4f}x+{params[1]:.4f}"
        elif name == "Exponential":
            eq_text = f"y={params[0]:.4f}e^({params[1]:.4f}x)"
        elif name == "Power-Law":
            eq_text = f"y={params[0]:.4f}x^{params[1]:.4f}"
        elif name == "Logarithmic":
            eq_text = f"y={params[0]:.4f}ln(x)+{params[1]:.4f}"
        elif name == "Polynomial_2":
            eq_text = f"y={params[0]:.4f}+{params[1]:.4f}x+{params[2]:.4f}x^2"
        else:
            eq_text = "Unknown"
        x_pos = np.percentile(x_vals, 80)
        y_pos = model_func(x_pos, *params) * 1.1  # Slight upward offset to avoid overlap
        plt.text(x_pos, y_pos, eq_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel("2007 LAI")
plt.ylabel("Monthly Increase in SOC (g/kg/month)")
plt.title("Empirical Fit of 2007 LAI vs. SOC Monthly Increase")
plt.legend()
plt.grid(True)
plt.show()
