import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# Load the CSV file
csv_file =DATA_DIR / "Vegetation_Input_v2_with_Lai.csv"
df = pd.read_csv(csv_file)

# Standardize column names (remove spaces)
df.columns = df.columns.str.strip()

# Define relevant columns
lai_col = "Lai"
soc_col = "Monthly Increase in Soil Organic Carbon (g/kg)"

# Print original SOC values (to check for formatting issues)
print("Original SOC values before cleaning:\n", df[soc_col].head(10))

# Function to clean numeric values (removes special characters like '−')
def clean_numeric(value):
    if isinstance(value, str):
        value = re.sub(r'[^\d\.-]', '', value)  # Keep only numbers, dots, and minus
        try:
            return float(value) if value else np.nan
        except ValueError:
            return np.nan
    return value

# Apply cleaning function
df[soc_col] = df[soc_col].apply(clean_numeric)
df[lai_col] = df[lai_col].apply(clean_numeric)

# Print cleaned SOC values to verify correct conversion
print("Cleaned SOC values:\n", df[soc_col].head(10))

# Remove NaN values (but keep zeros and negatives)
df_cleaned = df.dropna(subset=[lai_col, soc_col])

# Check number of negative values
negative_points = df_cleaned[df_cleaned[soc_col] < 0]
positive_points = df_cleaned[df_cleaned[soc_col] > 0]
print(f"Total data points: {len(df_cleaned)}")
print(f"Positive SOC points: {len(positive_points)}")
print(f"Zero or negative SOC points: {len(negative_points)}")

# Extract X and Y
X = df_cleaned[lai_col].values
Y = df_cleaned[soc_col].values

### Define Empirical Models ###
# Linear Model: y = a * LAI + b
def linear_model(x, a, b):
    return a * x + b

# Exponential Model: y = a * exp(b * LAI)
def exponential_model(x, a, b):
    return a * np.exp(b * x)

# Power Law Model: y = a * LAI^b
def power_model(x, a, b):
    return a * np.power(x, b)

# Logarithmic Model: y = a * log(LAI) + b
def logarithmic_model(x, a, b):
    return a * np.log(x) + b

### Fit Each Model ###
models = {
    "Linear": linear_model,
    "Exponential": exponential_model,
    "Power-Law": power_model,
    "Logarithmic": logarithmic_model
}

fit_results = {}

for name, model in models.items():
    try:
        params, _ = curve_fit(model, X, Y, maxfev=5000)
        fit_results[name] = params

        # Print equation correctly
        if name == "Linear":
            equation_text = f"y = {params[0]:.8f} * LAI + {params[1]:.8f}"
        elif name == "Exponential":
            equation_text = f"y = {params[0]:.8f} * exp({params[1]:.8f} * LAI)"
        elif name == "Power-Law":
            equation_text = f"y = {params[0]:.8f} * LAI^{params[1]:.8f}"
        elif name == "Logarithmic":
            equation_text = f"y = {params[0]:.8f} * log(LAI) + {params[1]:.8f}"

        print(f"{name} Model: {equation_text}")
    except Exception as e:
        print(f"Could not fit {name} model: {e}")

# Generate X values for plotting
x_vals = np.linspace(X.min(), X.max(), 100)

# Plot Data: Separate Positive and Negative SOC Values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=positive_points[lai_col], y=positive_points[soc_col], color="blue", alpha=0.7, label="Positive SOC")
sns.scatterplot(x=negative_points[lai_col], y=negative_points[soc_col], color="red", alpha=0.7, label="Negative SOC")

# Plot Fitted Models and Add Equations to the Graph
for name, model in models.items():
    if name in fit_results:
        params = fit_results[name]
        y_vals = model(x_vals, *params)
        plt.plot(x_vals, y_vals, label=f"{name} Fit")

        # Format equation text
        if name == "Linear":
            equation_text = f"y = {params[0]:.4f} * LAI + {params[1]:.4f}"
        elif name == "Exponential":
            equation_text = f"y = {params[0]:.4f} * exp({params[1]:.4f} * LAI)"
        elif name == "Power-Law":
            equation_text = f"y = {params[0]:.4f} * LAI^{params[1]:.4f}"
        elif name == "Logarithmic":
            equation_text = f"y = {params[0]:.4f} * log(LAI) + {params[1]:.4f}"

        # Position equation text dynamically
        x_pos = np.percentile(x_vals, 80)  # Place equation at 80% of x-range
        y_pos = model(x_pos, *params) * 1.1  # Adjust y position slightly above curve
        plt.text(x_pos, y_pos, equation_text, fontsize=10, color="black", bbox=dict(facecolor='white', alpha=0.5))

# Labels and Titles
plt.xlabel("Leaf Area Index (LAI)")
plt.ylabel("Monthly Increase in Soil Organic Carbon (g/kg)")
plt.title("Empirical Fit of LAI vs. SOC Increase (Including Negative Values)")
plt.legend()
plt.grid(True)

# Show Plot
plt.show()
