import geopandas as gpd
import matplotlib.pyplot as plt

# Load the shapefile
shapefile_path = "D:\EcoSci\Dr.Shi\Data\94分区\黄土高原产沙分区（河流-植被-地貌94区）.shp"  # Replace with your actual file path
gdf = gpd.read_file(shapefile_path)

# Plot the shapefile
fig, ax = plt.subplots(figsize=(10, 6))
gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue")

# Customize the visualization
ax.set_title("Shapefile Visualization", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True)

# Show the plot
plt.show()
