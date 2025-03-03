import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Define your TIFF file path
tiff_file = r"D:\EcoSci\Dr.Shi\Data\k1_halfDegree.tif"  # Use 'r' before path to avoid escape characters

# Open the TIFF file
with rasterio.open(tiff_file) as dataset:
    band1 = dataset.read(1)  # Read the first band

    # Print metadata
    print(f"TIFF Metadata:")
    print(f"Width: {dataset.width}, Height: {dataset.height}")
    print(f"Number of Bands: {dataset.count}")
    print(f"Coordinate Reference System (CRS): {dataset.crs}")
    print(f"Bounding Box: {dataset.bounds}")

    # Print value range
    print("\nData Range:")
    print(f"Min Value: {np.nanmin(band1)}")
    print(f"Max Value: {np.nanmax(band1)}")

    # Identify NoData value
    nodata_value = dataset.nodata
    print(f"NoData Value: {nodata_value}")

    # Mask NoData values
    if nodata_value is not None:
        band1 = np.where(band1 == nodata_value, np.nan, band1)

    # Handle extreme values (set reasonable limits)
    valid_min, valid_max = np.nanpercentile(band1, [2, 98])  # Get 2nd and 98th percentiles
    print(f"Valid Data Range: {valid_min} to {valid_max}")

    # Clip extreme values for visualization
    band1_clipped = np.clip(band1, valid_min, valid_max)

    # Optional: Apply logarithmic scaling to handle extreme variations
    band1_log = np.log1p(np.maximum(0, band1_clipped))  # Avoid log(0) issues

    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Normal Visualization
    img1 = ax[0].imshow(band1_clipped, cmap='viridis', interpolation='nearest')
    ax[0].set_title("TIFF Visualization (Clipped)")
    ax[0].set_xlabel("X Coordinate")
    ax[0].set_ylabel("Y Coordinate")
    fig.colorbar(img1, ax=ax[0], label="Pixel Value")

    # Logarithmic Visualization
    img2 = ax[1].imshow(band1_log, cmap='plasma', interpolation='nearest')
    ax[1].set_title("Log-Scaled TIFF Visualization")
    ax[1].set_xlabel("X Coordinate")
    ax[1].set_ylabel("Y Coordinate")
    fig.colorbar(img2, ax=ax[1], label="Log(1+Value)")

    plt.show()
