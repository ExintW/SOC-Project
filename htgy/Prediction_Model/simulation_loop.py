import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import Paths 
from global_structs import MAP_STATS, INIT_VALUES
from config import *

def run_simulation_year(year, past=False):
    print(f"Running Simulation year {year}...")