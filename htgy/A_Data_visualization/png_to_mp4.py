import cv2
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

def generate_mp4(start_year, end_year):

    input_dir = OUTPUT_DIR / 'Figure'
    output_video = OUTPUT_DIR / f"SOC_{start_year}_{end_year}.mp4"

    start_time = time.time()

    # Frame size will be determined from the first image
    first_img = cv2.imread(str(input_dir / f"SOC_{start_year}_01_River.png"))
    if first_img is None:
        raise FileNotFoundError("First frame not found. Check the path and filename.")
    height, width, _ = first_img.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec compatible with .mp4
    fps = 20  # frames per second (adjust for speed)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Add frames
    for year in range(start_year, end_year+1):
        for month in range(1, 13):
            filename = f"SOC_{year}_{month:02d}_River.png"
            filepath = input_dir / filename
            if filepath.exists():
                frame = cv2.imread(str(filepath))
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                video_writer.write(frame)
            else:
                print(f"Missing: {filename}")

    video_writer.release()

    end_time = time.time()
    print(f"Video saved to {output_video}")
    print(f"Took: {(end_time-start_time):02f}s")
    
if __name__ == "__main__":
    start_year = 1980
    end_year = 2009
    generate_mp4(start_year, end_year)