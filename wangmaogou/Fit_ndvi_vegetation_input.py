import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# Define the exponential function with an added constant c: y = a * exp(b * x) + c
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Load the CSV file
file_path = r"D:\EcoSci\Dr.Shi\Data\Vegetation_Input.csv"
df = pd.read_csv(file_path)

# Extract NDVI and Daily SOC Increase columns
ndvi = df['NDVI'].values
daily_soc_increase = df['Daily Increase in Soil Organic Carbon (g/kg)'].values

# Remove invalid values where daily SOC increase ≤ 0 (if necessary)
valid_indices = daily_soc_increase > 0
ndvi = ndvi[valid_indices]
daily_soc_increase = daily_soc_increase[valid_indices]

# Initial guess for parameters: a, b, c
initial_guess = [1, 0.01, 0]

# Fit the exponential function to the data
params, covariance = curve_fit(exponential_func, ndvi, daily_soc_increase, p0=initial_guess)
a, b, c = params

# Generate predictions for plotting
ndvi_range = np.linspace(ndvi.min(), ndvi.max(), 100)
predicted_soc = exponential_func(ndvi_range, a, b, c)

# Create the regression equation text
equation_text = f"y = {a:.8f} * e^({b:.8f} * x) + {c:.8f}"

# Plot the data and the fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(ndvi, daily_soc_increase, color='blue', label='Data Points')
plt.plot(ndvi_range, predicted_soc, color='red', linewidth=2, label='Fitted Exponential Curve')
plt.xlabel('NDVI')
plt.ylabel('Daily Increase in Soil Organic Carbon (g/kg)')
plt.title('Exponential Fit (y = a * e^(b * x) + c): NDVI vs. Daily Increase in SOC')
plt.legend()

# Display the regression equation on the plot
plt.text(0.05, 0.95, equation_text, fontsize=12, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.grid(True)
plt.show()

# Print the regression equation
print(f"Regression equation: {equation_text}")

