import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import re
import os
import sys
from htgy.globals import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



# -----------------------------------------------------------
# 1. Read the new Excel file containing all 5 sheets
# -----------------------------------------------------------
excel_file = PROCESSED_DIR / "River_Basin_Points_Updated.xlsx"

# Read all sheets and merge into a single DataFrame
xls = pd.ExcelFile(excel_file)
all_sheets = xls.sheet_names

df_list = []
for sheet in all_sheets:
    df_sheet = pd.read_excel(xls, sheet_name=sheet)
    df_sheet["SheetName"] = sheet  # Optional: record which sheet the row came from
    df_list.append(df_sheet)

df = pd.concat(df_list, ignore_index=True)

# Standardize column names by stripping any extra spaces
df.columns = df.columns.str.strip()

# -----------------------------------------------------------
# 2. Define relevant columns and clean numeric values if needed
# -----------------------------------------------------------
# We use "2007 LAI" as the independent variable (X) and
# "SOC Monthly Increase (g/kg/month)" as the dependent variable (Y).
lai_col = "2007 LAI"
soc_col = "SOC Monthly Increase (g/kg/month)"

print("Preview of selected columns:")
print(df[[lai_col, soc_col]].head())


def clean_numeric(value):
    if isinstance(value, str):
        # Remove any characters except digits, period, and minus sign
        value = re.sub(r'[^\d\.-]', '', value)
        try:
            return float(value) if value else np.nan
        except ValueError:
            return np.nan
    return value


df[soc_col] = df[soc_col].apply(clean_numeric)
df[lai_col] = df[lai_col].apply(clean_numeric)

# Remove rows with NaN in either column
df_cleaned = df.dropna(subset=[lai_col, soc_col])

# -----------------------------------------------------------
# 3. Separate data points for plotting (optional)
# -----------------------------------------------------------
negative_points = df_cleaned[df_cleaned[soc_col] < 0]
positive_points = df_cleaned[df_cleaned[soc_col] > 0]
print(f"Total data points: {len(df_cleaned)}")
print(f"Positive SOC Increase points: {len(positive_points)}")
print(f"Zero or negative SOC Increase points: {len(negative_points)}")

# Extract arrays for curve fitting
X = df_cleaned[lai_col].values
Y = df_cleaned[soc_col].values


# -----------------------------------------------------------
# 4. Define empirical models
# -----------------------------------------------------------
def linear_model(x, a, b):
    return a * x + b


def exponential_model(x, a, b):
    return a * np.exp(b * x)


def power_model(x, a, b):
    return a * np.power(x, b)


def logarithmic_model(x, a, b):
    return a * np.log(x) + b


models = {
    "Linear": linear_model,
    "Exponential": exponential_model,
    "Power-Law": power_model,
    "Logarithmic": logarithmic_model
}

fit_results = {}

# -----------------------------------------------------------
# 5. Fit each model using curve_fit
# -----------------------------------------------------------
for name, model in models.items():
    try:
        params, _ = curve_fit(model, X, Y, maxfev=5000)
        fit_results[name] = params
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

# -----------------------------------------------------------
# 6. Plotting the data and fitted models
# -----------------------------------------------------------
x_vals = np.linspace(X.min(), X.max(), 100)

plt.figure(figsize=(10, 6))

# Plot scatter points for positive and negative SOC monthly increase
sns.scatterplot(x=positive_points[lai_col], y=positive_points[soc_col],
                color="blue", alpha=0.7, label="Positive SOC Increase")
sns.scatterplot(x=negative_points[lai_col], y=negative_points[soc_col],
                color="red", alpha=0.7, label="Negative SOC Increase")

# Plot fitted curves and annotate equations
for name, model in models.items():
    if name in fit_results:
        params = fit_results[name]
        y_vals = model(x_vals, *params)
        plt.plot(x_vals, y_vals, label=f"{name} Fit")
        # Format the equation text for annotation
        if name == "Linear":
            equation_text = f"y = {params[0]:.4f} * LAI + {params[1]:.4f}"
        elif name == "Exponential":
            equation_text = f"y = {params[0]:.4f} * exp({params[1]:.4f} * LAI)"
        elif name == "Power-Law":
            equation_text = f"y = {params[0]:.4f} * LAI^{params[1]:.4f}"
        elif name == "Logarithmic":
            equation_text = f"y = {params[0]:.4f} * log(LAI) + {params[1]:.4f}"

        # Dynamically position the equation annotation
        x_pos = np.percentile(x_vals, 10)
        y_pos = model(x_pos, *params) * 1.1  # slightly above the curve
        plt.text(x_pos, y_pos, equation_text, fontsize=10, color="black",
                 bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel("2007 LAI")
plt.ylabel("SOC Monthly Increase (g/kg/month)")
plt.title("Empirical Fit of 2007 LAI vs. SOC Monthly Increase (Combined Data)")
plt.legend()
plt.grid(True)
plt.show()
