import os
import pandas as pd

def parquet_to_csv(input_path: str, output_path: str = None) -> None:
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + '.csv'

    df = pd.read_parquet(input_path)
    df.to_csv(output_path, index=False)
    print(f"Converted '{input_path}' → '{output_path}'")


if __name__ == "__main__":
    # Your specific paths
    input_file = r"D:\EcoSci\Dr.Shi\SOC_Github\Output\Data\SOC_Present\SOC_terms_2007_01_River.parquet"
    output_dir = r"D:\EcoSci\Dr.Shi\SOC_Github\Output\Data\SOC_Present"

    # Build the output filename by replacing the .parquet suffix with .csv
    filename = os.path.basename(input_file)                    # "SOC_terms_1971_06_River.parquet"
    base, _   = os.path.splitext(filename)                     # ("SOC_terms_1971_06_River", ".parquet")
    output_file = os.path.join(output_dir, base + ".csv")      # ".../SOC_terms_1971_06_River.csv"

    parquet_to_csv(input_file, output_file)
