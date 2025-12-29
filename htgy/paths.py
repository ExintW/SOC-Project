from pathlib import Path

class Paths:
    WORKING_DIR = Path(__file__).parent.parent
    DATA_DIR = WORKING_DIR / "Raw_Data"
    PROCESSED_DIR = WORKING_DIR / "Processed_Data"
    OUTPUT_DIR = WORKING_DIR / "Output"