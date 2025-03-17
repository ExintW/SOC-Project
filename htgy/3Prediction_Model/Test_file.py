import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from pathlib import Path
import sys

# Append globals (assumes DATA_DIR, PROCESSED_DIR, OUTPUT_DIR are defined in globals.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

river_shp = gpd.read_file(DATA_DIR / "China_River" / "ChinaRiver_main.shp")
small_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "htgy_River_Basin.shp")
large_boundary_shp = gpd.read_file(DATA_DIR / "River_Basin" / "94_area.shp")

print("River CRS:", river_shp.crs)
print("Small Boundary CRS:", small_boundary_shp.crs)
print("Large Boundary CRS:", large_boundary_shp.crs)

