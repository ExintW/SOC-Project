from PIL import Image
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

start_year = 1950
end_year = 2009

input_dir = OUTPUT_DIR / 'Figure' / "Tikhonov freq 5 LAI trend 0.1A"  
output_gif = OUTPUT_DIR / f"SOC_{start_year}_{end_year}.gif"

start_time = time.time()

image_files = []
for year in range(start_year, end_year+1):
    for month in range(1, 13):
        filename = f"SOC_{year}_{month:02d}_River.png"
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            image_files.append(filepath)
        else:
            print(f"Warning: Missing file {filepath}")

# Load images
frames = [Image.open(f).convert("RGBA") for f in image_files]

# Save as animated GIF
if frames:
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=50,  # milliseconds per frame
        loop=0
    )
    end_time = time.time()
    print(f"GIF saved to {output_gif}")
    print(f"Took: {end_time-start_time:02f}s")
else:
    print("No images found to generate GIF.")
