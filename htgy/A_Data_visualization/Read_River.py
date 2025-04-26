import geopandas as gpd
import matplotlib.pyplot as plt

# Define the shapefile path
shapefile_path = r"D:\EcoSci\Dr.Shi\Data\China_River\China_River_smallest.shp"

# Read the shapefile
try:
    gdf = gpd.read_file(shapefile_path)

    # Check if the file contains data
    if gdf.empty:
        print("The shapefile is empty.")
    else:
        # ---- 🎨 VISUALIZATION ----
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the rivers with a blue color and black edge
        gdf.plot(ax=ax, color="blue", edgecolor="black", linewidth=0.5)

        # Customize visualization
        ax.set_title("Visualization of China Rivers", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.grid(True)
        plt.show()

except Exception as e:
    print(f"Error reading the shapefile: {e}")
