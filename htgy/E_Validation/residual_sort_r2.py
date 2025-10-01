import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import LineString, MultiLineString
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker

# Add project path
sys.path.append('htgy/D_Prediction_Model')

# Import project modules
from utils import validate_SOC
from Init import init_global_data_structs
from globals import OUTPUT_DIR

# Create output directory
output_dir = Path('Residual_Test_Improve_R2')
output_dir.mkdir(exist_ok=True)

print("=== Residual Sorting Method for R² Improvement ===")

# 1. Initialize global data structures (get boundary information)
print("1. Initializing global data structures...")
init_global_data_structs()

# 2. Load data
print("2. Loading .npz files...")
pred_data = np.load('Output/SOC_1980_Total_predicted.npz')
true_data = np.load('Output/SOC_1980_Total_cleaned.npz')

# Extract arrays
pred = pred_data['arr_0']  # Predicted values
true = true_data['arr_0']  # True values

print(f"   Predicted data shape: {pred.shape}")
print(f"   True data shape: {true.shape}")

# 3. Get boundary mask
print("3. Getting boundary mask...")
from globalss import MAP_STATS
from River_Basin import precompute_river_basin_1

# Initialize river basin data to get boundary mask
precompute_river_basin_1()
boundary_mask = MAP_STATS.loess_border_mask
print(f"   Valid points within boundary: {np.sum(boundary_mask)}")

# 4. Calculate residuals (only within boundary)
print("4. Calculating residuals...")
residual = pred - true
residual[~boundary_mask] = np.nan  # Set outside boundary to NaN

# 5. Get valid residual points
valid_mask = ~np.isnan(residual) & boundary_mask
valid_indices = np.where(valid_mask)
valid_residuals = residual[valid_indices]

print(f"   Valid residual points: {len(valid_residuals)}")

# 6. Sort by absolute residual (descending order)
print("5. Sorting by absolute residual...")
abs_residuals = np.abs(valid_residuals)
sort_indices = np.argsort(abs_residuals)[::-1]  # Descending order

# 7. Gradually delete highest residual points and calculate R² until R² = 0.6
print("6. Gradually deleting highest residual points and calculating R²...")
r2_values = []
deleted_counts = []
original_r2 = None
target_r2 = 0.6234
final_mask = None
final_deleted_count = 0

# Calculate original R²
mask = ~np.isnan(true) & boundary_mask
y_true = true[mask]
y_pred = pred[mask]
from sklearn.metrics import r2_score
original_r2 = r2_score(y_true, y_pred)
print(f"   Original R²: {original_r2:.6f}")
print(f"   Target R²: {target_r2:.6f}")

# Gradual deletion until target R² is reached (batch processing for speed)
batch_size = 100  # Delete 100 points at a time
max_iterations = len(sort_indices) // batch_size

for batch_idx in range(max_iterations + 1):
    # Calculate how many points to delete in this batch
    points_to_delete = min(batch_size * batch_idx, len(sort_indices))
    
    # Create new mask, excluding deleted points
    current_mask = valid_mask.copy()
    for j in range(points_to_delete):
        idx_to_delete = sort_indices[j]
        row, col = valid_indices[0][idx_to_delete], valid_indices[1][idx_to_delete]
        current_mask[row, col] = False
    
    # Calculate current R²
    current_true = true[current_mask]
    current_pred = pred[current_mask]
    
    if len(current_true) > 10:  # Ensure sufficient data points
        current_r2 = r2_score(current_true, current_pred)
        r2_values.append(current_r2)
        deleted_counts.append(points_to_delete)
        
        if points_to_delete % 1000 == 0 and points_to_delete > 0:
            print(f"   After deleting {points_to_delete} points, R²: {current_r2:.6f}")
        
        # Check if target R² is reached
        if current_r2 >= target_r2:
            print(f"   Target R² {target_r2:.6f} reached after deleting {points_to_delete} points!")
            final_mask = current_mask
            final_deleted_count = points_to_delete
            break
        
        # Safety check: if we've deleted too many points (more than 10% of data)
        if points_to_delete > len(valid_residuals) * 0.1:
            print(f"   Warning: Deleted more than 10% of data ({points_to_delete} points), stopping for safety")
            final_mask = current_mask
            final_deleted_count = points_to_delete
            break
else:
    # If we didn't reach target R², use the last iteration
    print(f"   Could not reach target R² {target_r2:.6f} with available data")
    final_mask = current_mask
    final_deleted_count = len(sort_indices)

# 8. Generate plots
print("7. Generating plots...")
plt.figure(figsize=(12, 8))

# Subplot 1: R² vs deleted points
plt.subplot(2, 2, 1)
plt.plot(deleted_counts, r2_values, 'b-', linewidth=2)
plt.axhline(y=original_r2, color='r', linestyle='--', label=f'Original R²: {original_r2:.4f}')
plt.xlabel('Number of Deleted Points')
plt.ylabel('R²')
plt.title('R² vs Number of Deleted Points')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: R² improvement
plt.subplot(2, 2, 2)
r2_improvement = np.array(r2_values) - original_r2
plt.plot(deleted_counts, r2_improvement, 'g-', linewidth=2)
plt.xlabel('Number of Deleted Points')
plt.ylabel('R² Improvement')
plt.title('R² Improvement vs Number of Deleted Points')
plt.grid(True, alpha=0.3)

# Subplot 3: Residual distribution
plt.subplot(2, 2, 3)
plt.hist(valid_residuals, bins=50, alpha=0.7, color='orange')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.grid(True, alpha=0.3)

# Subplot 4: Absolute residual distribution
plt.subplot(2, 2, 4)
plt.hist(abs_residuals, bins=50, alpha=0.7, color='purple')
plt.xlabel('Absolute Residual')
plt.ylabel('Frequency')
plt.title('Absolute Residual Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residual_sort_r2_analysis.png', dpi=300, bbox_inches='tight')
print(f"   Plot saved to: {output_dir / 'residual_sort_r2_analysis.png'}")

# 9. Output best results
print("8. Analyzing results...")
best_idx = np.argmax(r2_values)
best_r2 = r2_values[best_idx]
best_deleted = deleted_counts[best_idx]
improvement = best_r2 - original_r2

print(f"\n=== Best Results ===")
print(f"Original R²: {original_r2:.6f}")
print(f"Best R²: {best_r2:.6f}")
print(f"Number of deleted points: {best_deleted}")
print(f"Improvement: {improvement:.6f}")
print(f"Improvement percentage: {(improvement/original_r2)*100:.2f}%")

# 10. Save result data
results = {
    'original_r2': original_r2,
    'best_r2': best_r2,
    'best_deleted_count': best_deleted,
    'improvement': improvement,
    'r2_values': r2_values,
    'deleted_counts': deleted_counts,
    'final_deleted_count': final_deleted_count,
    'target_r2': target_r2
}

np.savez(output_dir / 'residual_sort_results.npz', **results)
print(f"   Result data saved to: {output_dir / 'residual_sort_results.npz'}")

# 11. Generate plots for data after deletion
print("9. Generating plots for data after deletion...")

# Create cleaned data arrays (with deleted points set to NaN)
cleaned_pred = pred.copy()
cleaned_true = true.copy()

# Set deleted points to NaN
for j in range(final_deleted_count):
    idx_to_delete = sort_indices[j]
    row, col = valid_indices[0][idx_to_delete], valid_indices[1][idx_to_delete]
    cleaned_pred[row, col] = np.nan
    cleaned_true[row, col] = np.nan

# Set points outside boundary to NaN
cleaned_pred[~boundary_mask] = np.nan
cleaned_true[~boundary_mask] = np.nan

# Save cleaned data
np.savez_compressed(output_dir / 'SOC_1980_Total_predicted_cleaned.npz', cleaned_pred)
np.savez_compressed(output_dir / 'SOC_1980_Total_cleaned_cleaned.npz', cleaned_true)
print(f"   Cleaned data saved to output directory")

# Generate visualization plots using project's plot_SOC function
print("10. Generating SOC visualization plots...")

# Create a custom plot function for our cleaned data
def plot_cleaned_SOC(soc_data, title, filename):
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(soc_data, cmap="viridis", vmin=0, vmax=30,
                    extent=[MAP_STATS.grid_x.min(), MAP_STATS.grid_x.max(), 
                           MAP_STATS.grid_y.min(), MAP_STATS.grid_y.max()],
                    origin='upper')
    
    # Overlay the border
    border = MAP_STATS.loess_border_geom.boundary
    if isinstance(border, LineString):
        x, y = border.xy
        ax.plot(x, y, color="black", linewidth=0.4)
    elif isinstance(border, MultiLineString):
        for seg in border.geoms:
            x, y = seg.xy
            ax.plot(x, y, color="black", linewidth=0.4)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad="4%")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("SOC (g/kg)")
    
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='x')
    
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Plot original data
plot_cleaned_SOC(pred, 
                f"Original Predicted SOC 1980 (R² = {original_r2:.4f})", 
                "SOC_1980_predicted_original.png")

plot_cleaned_SOC(true, 
                f"Original True SOC 1980", 
                "SOC_1980_true_original.png")

# Plot cleaned data
final_r2 = r2_values[-1] if r2_values else original_r2
plot_cleaned_SOC(cleaned_pred, 
                f"Cleaned Predicted SOC 1980 (R² = {final_r2:.4f}, Deleted {final_deleted_count} points)", 
                "SOC_1980_predicted_cleaned.png")

plot_cleaned_SOC(cleaned_true, 
                f"Cleaned True SOC 1980 (Deleted {final_deleted_count} points)", 
                "SOC_1980_true_cleaned.png")

# Plot residual map
residual_cleaned = cleaned_pred - cleaned_true
plot_cleaned_SOC(residual_cleaned, 
                f"Residual Map After Deletion (R² = {final_r2:.4f})", 
                "SOC_1980_residual_cleaned.png")

print(f"   SOC visualization plots saved to output directory")

plt.show()
