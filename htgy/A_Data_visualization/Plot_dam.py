import sys, os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ─── 1) PROJECT SETUP ─────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import PROCESSED_DIR, OUTPUT_DIR  # Path objects

# ─── 2) FILE PATHS ────────────────────────────────────────────────────────────
dam_csv_path = PROCESSED_DIR / "htgy_dams_fixed.csv"
out_csv      = OUTPUT_DIR    / "dam_storage_by_year.csv"
out_png      = OUTPUT_DIR    / "dam_storage_by_year.png"

out_csv.parent.mkdir(parents=True, exist_ok=True)
out_png.parent.mkdir(parents=True, exist_ok=True)

# ─── 3) LOAD & CLEAN YEAR ─────────────────────────────────────────────────────
df = pd.read_csv(dam_csv_path)

# convert any floats like 1995.4 → 1995, 1995.9 → 1995
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)

# ─── 4) SUM STORAGE PER INTEGER YEAR ─────────────────────────────────────────
storage_by_year = (
    df
    .groupby("year", as_index=False)["total_stor"]
    .sum()
    .rename(columns={"total_stor": "total_storage_10k_m3"})
    .sort_values("year")
)

# ─── 5) SAVE CSV ──────────────────────────────────────────────────────────────
storage_by_year.to_csv(out_csv, index=False)
print(f"Saved annual storage per year to: {out_csv}")

# ─── 6) PLOT ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(
    storage_by_year["year"],
    storage_by_year["total_storage_10k_m3"],
    marker="o",
    linestyle="-",
    linewidth=1.5
)
plt.xlabel("Build Year")
plt.ylabel("Total Dam Storage (10 000 m³)")
plt.title("Annual Dam Storage by Build Year")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# ─── 7) SAVE FIGURE ──────────────────────────────────────────────────────────
plt.savefig(out_png, dpi=300)
print(f"Saved plot to: {out_png}")
plt.show()
