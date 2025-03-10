import geopandas as gpd
import matplotlib.pyplot as plt

# File paths for the shapefiles
basin_fp = r"D:\EcoSci\Dr.Shi\Data\htgy_River_Basin\htgy_River_Basin.shp"
river_fp = r"D:\EcoSci\Dr.Shi\Data\China_River\China_River_smallest.shp"

# Read the shapefiles
basin = gpd.read_file(basin_fp)
river = gpd.read_file(river_fp)

# Check the coordinate reference systems (CRS) and reproject if necessary
print("Basin CRS:", basin.crs)
print("River CRS:", river.crs)
if basin.crs != river.crs:
    river = river.to_crs(basin.crs)
    print("Reprojected river layer to match basin CRS.")

# Clip the river features using the river basin border as the mask
river_clipped = gpd.clip(river, basin)

# Plotting only the river basin region
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the basin border
basin.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, label='Basin Border')
# Plot the clipped river (only within the basin)
river_clipped.plot(ax=ax, color='red', linewidth=0.5, label='River within Basin')

# Set the axis limits to the basin's extent
minx, miny, maxx, maxy = basin.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

plt.title("River within Basin Region")
plt.legend()
plt.show()

# Optionally, save the clipped river shapefile to a new file
river_clipped.to_file(r"D:\EcoSci\Dr.Shi\Data\river_within_basin.shp")
