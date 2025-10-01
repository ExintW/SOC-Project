import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # Expects DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

def parquet_to_csv(input_path: str, output_path: str = None) -> None:
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + '.csv'

    df = pd.read_parquet(input_path)
    df.to_csv(output_path, index=False)
    print(f"Converted '{input_path}' → '{output_path}'")


if __name__ == "__main__":
    # Your specific paths
    input_file = OUTPUT_DIR / "Data" / "SOC_terms_2025_02_River.parquet"
    #output_dir = OUTPUT_DIR / "Data" / "SOC_present 5"

    #input_file = OUTPUT_DIR / "Data" / "SOC_Past 2" /  "SOC_terms_1980_01_River.parquet"
    output_dir = OUTPUT_DIR / "Data"

    # Build the output filename by replacing the .parquet suffix with .csv
    filename = os.path.basename(input_file)                    # "SOC_terms_1971_06_River.parquet"
    base, _   = os.path.splitext(filename)                     # ("SOC_terms_1971_06_River", ".parquet")
    output_file = os.path.join(output_dir, base + ".csv")      # ".../SOC_terms_1971_06_River.csv"

    parquet_to_csv(input_file, output_file)
