import geopandas as gpd
import matplotlib.pyplot as plt

# Define base path
base_path = r"D:\EcoSci\Dr.Shi\Data"

# Path to your shapefile
shapefile_path = f"{base_path}\\骨干坝\\骨干坝.shp"

# Read the shapefile
gdf = gpd.read_file(shapefile_path)

# Save to CSV with UTF-8 encoding (fixing garbled Chinese characters)
output_csv_path = f"{base_path}\\htgy_dams_fixed.csv"
gdf.drop(columns="geometry").to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"Shapefile data saved to {output_csv_path} with UTF-8 encoding.")

# ---- 🎨 VISUALIZATION ----
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the shapefile
gdf.plot(ax=ax, color="blue", edgecolor="black", alpha=0.6)

# Customize visualization
ax.set_title("Visualization of 骨干坝 Shapefile", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.grid(True)
plt.show()
gdf = gpd.read_file(shapefile_path)

# Save to CSV with UTF-8 encoding (fixing garbled Chinese characters)
output_csv_path = f"{base_path}\\htgy_dams_fixed.csv"
gdf.drop(columns="geometry").to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"Shapefile data saved to {output_csv_path} with UTF-8 encoding.")
