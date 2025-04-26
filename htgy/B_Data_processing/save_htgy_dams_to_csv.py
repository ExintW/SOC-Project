import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

# Paths to shapefiles
shapefile_paths = {
    "骨干坝": f"{DATA_DIR}\\骨干坝\\骨干坝.shp",  # "Backbone Dams"
    "中型坝": f"{DATA_DIR}\\中型坝\\中型坝.shp"  # "Medium-sized Dams"
}

# English translations for legend
legend_labels = {
    "骨干坝": "Backbone Dams",
    "中型坝": "Medium-sized Dams"
}

# List to store GeoDataFrames
gdfs = []

# ---- Data Processing ----
for name, path in shapefile_paths.items():
    gdf = gpd.read_file(path)
    if not gdf.empty:
        gdf["dam_type"] = legend_labels[name]  # Add a new column with English name
        gdfs.append(gdf.drop(columns="geometry", errors="ignore"))  # Drop geometry for CSV storage

# Merge all data into a single DataFrame
if gdfs:
    merged_df = pd.concat(gdfs, ignore_index=True)

    # Save to CSV with UTF-8 encoding
    output_csv_path = f"{PROCESSED_DIR}\\htgy_dams_fixed.csv"
    merged_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Shapefile data merged and saved to {output_csv_path} with UTF-8 encoding.")

    # ---- 🎨 VISUALIZATION ----
    fig, ax = plt.subplots(figsize=(10, 6))

    # Read shapefiles again for plotting
    for name, path in shapefile_paths.items():
        gdf = gpd.read_file(path)
        if not gdf.empty:
            gdf.plot(ax=ax, edgecolor="black", alpha=0.6, label=legend_labels[name])  # Use English labels

    # Customize visualization
    ax.set_title("Visualization of Backbone Dams & Medium-sized Dams", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    plt.grid(True)
    plt.show()
else:
    print("Error: No valid data found in the provided shapefiles.")
