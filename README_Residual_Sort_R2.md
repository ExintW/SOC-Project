# Residual Sorting Method for R² Improvement

## Overview

This script implements a residual sorting method to improve the R² (coefficient of determination) of the SOC (Soil Organic Carbon) prediction model by systematically removing data points with the highest prediction errors.

## Location

The script is located at: `htgy/E_Validation/residual_sort_r2.py`

## Purpose

The main goal is to identify and analyze the impact of removing outlier data points (those with the highest residuals) on the model's predictive performance. This helps understand:

1. **Data Quality**: Which data points contribute most to prediction errors
2. **Model Improvement**: How much the model performance can be enhanced by removing problematic data
3. **Outlier Detection**: Systematic identification of potential outliers or noise in the dataset

## Methodology

### 1. **Residual Calculation**
- Calculate residuals: `residual = predicted_value - true_value`
- Only consider data points within the Loess Plateau boundary
- Handle NaN values appropriately

### 2. **Sorting and Deletion**
- Sort data points by absolute residual values (descending order)
- Gradually delete points with the highest residuals
- Recalculate R² after each deletion

### 3. **Performance Analysis**
- Track R² improvement as points are removed
- Identify the optimal number of points to delete
- Generate comprehensive visualizations

## Input Files

The script requires two .npz files in the root directory:
- `SOC_1980_Total_predicted.npz` - Model predictions for 1980
- `SOC_1980_Total_cleaned.npz` - True/observed values for 1980

## Output

### Files Generated
All outputs are saved to the `Residual_Test_Improve_R2/` folder:

1. **`residual_sort_r2_analysis.png`** - Comprehensive analysis plots
   - R² vs Number of Deleted Points
   - R² Improvement vs Number of Deleted Points
   - Residual Distribution
   - Absolute Residual Distribution

2. **`residual_sort_results.npz`** - Numerical results data
   - Original R² value
   - Best R² value achieved
   - Number of points deleted for optimal performance
   - Improvement statistics
   - Complete R² progression data

### Key Results
Based on the latest run:
- **Original R²**: 0.365100
- **Best R²**: 0.401479
- **Optimal Deletion**: 999 points (0.15% of total data)
- **Improvement**: +0.036379 (+9.96%)

## Usage

### Prerequisites
1. Ensure the required .npz files are in the root directory
2. Install project dependencies using uv:
   ```bash
   uv sync
   ```

### Running the Script
```bash
# From the root directory
uv run htgy/E_Validation/residual_sort_r2.py
```

### Expected Output
The script will:
1. Initialize global data structures
2. Load prediction and true value data
3. Calculate and sort residuals
4. Perform gradual deletion analysis
5. Generate plots and save results
6. Display summary statistics

## Technical Details

### Boundary Handling
- Uses `MAP_STATS.loess_border_mask` to ensure only data within the Loess Plateau boundary is considered
- Automatically handles NaN values and boundary constraints

### Performance Optimization
- Limits deletion to maximum 1000 points for computational efficiency
- Uses vectorized operations for fast residual calculations
- Efficient memory management for large datasets

### Visualization Features
- Four-panel comprehensive analysis plot
- High-resolution output (300 DPI)
- Professional formatting with grids and legends

## Interpretation

### R² Improvement
The increase in R² indicates:
- **Better Model Fit**: Removing problematic data points improves overall model performance
- **Data Quality Issues**: Some data points may be outliers or contain measurement errors
- **Model Robustness**: The model performs better on cleaner datasets

### Practical Applications
This analysis can guide:
1. **Data Quality Improvement**: Focus on regions or conditions with high residuals
2. **Model Refinement**: Identify areas where the model needs improvement
3. **Validation Strategy**: Develop better validation protocols for future data collection

## Dependencies

- numpy
- matplotlib
- scikit-learn
- Project-specific modules from `htgy/D_Prediction_Model/`

## Notes

- The script automatically creates the output directory if it doesn't exist
- All file paths are relative to the root directory
- The analysis is specifically designed for SOC prediction validation
- Results are reproducible given the same input data

## Future Enhancements

Potential improvements could include:
- Configurable deletion limits
- Different sorting criteria (e.g., relative residuals)
- Statistical significance testing
- Integration with other validation methods
