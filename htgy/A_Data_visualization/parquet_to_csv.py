import pandas as pd
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

def parquet_to_csv(parquet_file, csv_file=None):
    # Read the Parquet file
    df = pd.read_parquet(parquet_file)

    # Determine CSV filename if not provided
    if csv_file is None:
        csv_file = os.path.splitext(parquet_file)[0] + '.csv'

    # Write to CSV
    df.to_csv(csv_file, index=False)
    print(f"Converted {parquet_file} to {csv_file}")

if __name__ == "__main__":
    parquet_file = OUTPUT_DIR / "Data" / "No Spatial weighted avg lambda 1 w LAI" / "SOC_terms_1950_01_River.parquet"
    csv_file = OUTPUT_DIR / "SOC_1950_01.csv"
    
    parquet_to_csv(parquet_file, csv_file)
