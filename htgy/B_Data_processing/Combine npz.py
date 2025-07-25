# ============================================================
# Script to combine .npz files spanning different time ranges
# into unified dynamic_data.npz and static_data.npz files,
# using memory-mapped arrays and trimming to the 2007–2100 window.
# ------------------------------------------------------------
# Sections:
#   1) IMPORTS
#   2) FILE PATHS
#   3) FUNCTION: combine_npz_to_memmap()
#   4) COMBINE & TRIM DYNAMIC ARRAYS + PRINT SHAPES
#   5) LOAD STATIC MASKS
#   6) SAVE FINAL NPZs
#   7) CLEAN UP TEMP MEMMAP FILES
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *  # Expects DATA_DIR, PROCESSED_DIR, OUTPUT_DIR

# 2) FILE PATHS
# — Dynamic data components:
fast_soc_paths = [
    OUTPUT_DIR / "Fast SOC year 2007-2024.npz",
    OUTPUT_DIR / "Fast SOC year 2025-2100.npz"
]
slow_soc_paths = [
    OUTPUT_DIR / "Slow SOC year 2007-2024.npz",
    OUTPUT_DIR / "Slow SOC year 2025-2100.npz"
]
# Updated v_fast/v_slow: only 2007–2024 and 2025–2100 (ssp245)
v_fast_paths = [
    OUTPUT_DIR / "V_fast_2007-2024.npz",
    OUTPUT_DIR / "V_fast_2025-2100.npz"
]
v_slow_paths = [
    OUTPUT_DIR / "V_slow_2007-2024.npz",
    OUTPUT_DIR / "V_slow_2025-2100.npz"
]
precip_paths = [
    PROCESSED_DIR / "tp_1950-2024.npz",
    PROCESSED_DIR / "PR_245_2015-2100.npz"
]
dam_paths = [
    OUTPUT_DIR / "Active_dams_2007-2024.npz",
    OUTPUT_DIR / "Active_dams_2025-2100.npz"
]

# — Static data components:
dem_path  = OUTPUT_DIR / "DEM.npz"
mask_path = PROCESSED_DIR / "precomputed_masks.npz"

# — Temporary directory for memory-mapped files
tmp_dir = OUTPUT_DIR / "tmp_memmaps"
os.makedirs(tmp_dir, exist_ok=True)
tmp_fast_soc       = tmp_dir / "soc_fast_combined.npy"
tmp_slow_soc       = tmp_dir / "soc_slow_combined.npy"
tmp_v_fast_trimmed = tmp_dir / "v_fast_2007-2100.npy"
tmp_v_slow_trimmed = tmp_dir / "v_slow_2007-2100.npy"
tmp_precip_trimmed = tmp_dir / "precip_2007-2100.npy"
tmp_check_dams     = tmp_dir / "check_dams_combined.npy"

# 3) FUNCTION: combine_npz_to_memmap()
def combine_npz_to_memmap(source_paths, output_memmap_path, array_key=None):
    """
    Combine multiple .npz files (each containing one array) into a single
    memory-mapped .npy file. Avoids loading everything into RAM.

    Parameters:
        source_paths: list of Path to .npz files. Each must contain exactly
                      one 3D array [T_i, H, W].
        output_memmap_path: Path to the .npy file to create (memmap).
        array_key: Optional string key. If provided, uses this key to load
                   from each .npz; otherwise, uses the first key in each.

    Returns:
        combined_memmap: np.memmap of shape [(sum T_i), H, W].
    """
    total_time = 0
    height = width = None
    dtype = None
    archives = []
    keys = []
    times = []

    # First pass: inspect shapes & dtype, accumulate total_time
    for p in source_paths:
        archive = np.load(p, mmap_mode="r")
        key = array_key if array_key else archive.files[0]
        arr = archive[key]
        T_i, H_i, W_i = arr.shape

        if dtype is None:
            dtype = arr.dtype
            height, width = H_i, W_i
        else:
            if (H_i, W_i) != (height, width):
                raise ValueError(
                    f"Shape mismatch: {p} has shape {arr.shape},"
                    f" expected [*, {height}, {width}]."
                )
            if arr.dtype != dtype:
                raise ValueError(
                    f"Dtype mismatch: {p} dtype {arr.dtype}, expected {dtype}."
                )

        total_time += T_i
        archives.append(archive)
        keys.append(key)
        times.append(T_i)

    # Create memmap on disk
    combined_shape = (total_time, height, width)
    combined_memmap = np.memmap(
        output_memmap_path,
        dtype=dtype,
        mode="w+",
        shape=combined_shape
    )

    # Second pass: copy each source into the memmap
    cursor = 0
    for archive, key, T_i in zip(archives, keys, times):
        combined_memmap[cursor: cursor + T_i, :, :] = archive[key]
        cursor += T_i
        archive.close()

    return combined_memmap

# 4) COMBINE & TRIM DYNAMIC ARRAYS + PRINT SHAPES
print("Combining dynamic arrays (2007–2100) into memory‐mapped files...")

# 4a) soc_fast
soc_fast_memmap = combine_npz_to_memmap(fast_soc_paths, tmp_fast_soc)
print("soc_fast combined shape:", soc_fast_memmap.shape)

# 4b) soc_slow
soc_slow_memmap = combine_npz_to_memmap(slow_soc_paths, tmp_slow_soc)
print("soc_slow combined shape:", soc_slow_memmap.shape)

# 4c) v_fast (2007–2100)
v_fast_memmap = combine_npz_to_memmap(v_fast_paths, tmp_v_fast_trimmed)
print("v_fast combined shape:", v_fast_memmap.shape)

# 4d) v_slow (2007–2100)
v_slow_memmap = combine_npz_to_memmap(v_slow_paths, tmp_v_slow_trimmed)
print("v_slow combined shape:", v_slow_memmap.shape)

# 4e) precip (trim to 2007–2100)
print("Trimming precipitation to 2007–2100...")
tp = np.load(precip_paths[0], mmap_mode="r")
pr = np.load(precip_paths[1], mmap_mode="r")
key_tp = tp.files[0] if (tp := tp if False else tp) else tp # placeholder
# Actually load correctly:
key_tp = tp.files[0]
key_pr = pr.files[0]
arr_tp = tp[key_tp]
arr_pr = pr[key_pr]

months_2007_2100 = (2100 - 2007 + 1) * 12  # 1128 months
H, W = arr_tp.shape[1], arr_tp.shape[2]
precip_trimmed = np.memmap(
    tmp_precip_trimmed,
    dtype=arr_tp.dtype,
    mode="w+",
    shape=(months_2007_2100, H, W)
)
# Jan2007–Dec2024
start_idx_2007_tp = (2007 - 1950) * 12  # = 684
end_idx_2024_tp   = (2024 - 1950 + 1) * 12  # = 900
precip_trimmed[0:(end_idx_2024_tp - start_idx_2007_tp), :, :] = \
    arr_tp[start_idx_2007_tp:end_idx_2024_tp, :, :]
# Jan2025–Dec2100
start_idx_2025_pr = (2025 - 2015) * 12  # = 120
precip_trimmed[(end_idx_2024_tp - start_idx_2007_tp):, :, :] = \
    arr_pr[start_idx_2025_pr:(start_idx_2025_pr + (2100 - 2025 + 1)*12), :, :]
print("precip trimmed shape:", precip_trimmed.shape)
tp.close()
pr.close()

# 4f) check_dams
check_dams_memmap = combine_npz_to_memmap(dam_paths, tmp_check_dams)
print("check_dams combined shape:", check_dams_memmap.shape)

print("\nFinished combining & trimming dynamic arrays.\n")

# 5) LOAD STATIC MASKS
print("Loading static masks...")

dem_archive = np.load(dem_path)
dem_key = dem_archive.files[0]
dem = dem_archive[dem_key]
dem_archive.close()
print("dem shape:", dem.shape)

mask_archive = np.load(mask_path)
loess_border_mask    = mask_archive["loess_border_mask"]
river_mask           = mask_archive["river_mask"]
small_boundary_mask  = mask_archive["small_boundary_mask"]
large_boundary_mask  = mask_archive["large_boundary_mask"]
small_outlet_mask    = mask_archive["small_outlet_mask"]
large_outlet_mask    = mask_archive["large_outlet_mask"]
mask_archive.close()

print("loess_border_mask shape:", loess_border_mask.shape)
print("river_mask shape:       ", river_mask.shape)
print("small_boundary_mask shape:", small_boundary_mask.shape)
print("large_boundary_mask shape:", large_boundary_mask.shape)
print("small_outlet_mask shape:  ", small_outlet_mask.shape)
print("large_outlet_mask shape:  ", large_outlet_mask.shape)

print("\nFinished loading static data.\n")

# 6) SAVE FINAL NPZs
print("Saving dynamic_data.npz and static_data.npz...")

np.savez(
    OUTPUT_DIR / "dynamic_data.npz",
    soc_fast   = soc_fast_memmap,
    soc_slow   = soc_slow_memmap,
    v_fast     = v_fast_memmap,
    v_slow     = v_slow_memmap,
    precip     = precip_trimmed,
    check_dams = check_dams_memmap
)

np.savez(
    OUTPUT_DIR / "static_data.npz",
    dem                   = dem,
    loess_border_mask     = loess_border_mask,
    river_mask            = river_mask,
    small_boundary_mask   = small_boundary_mask,
    large_boundary_mask   = large_boundary_mask,
    small_outlet_mask     = small_outlet_mask,
    large_outlet_mask     = large_outlet_mask
)

print("Successfully saved both NPZ files.\n")

# 7) CLEAN UP TEMP MEMMAP FILES
for tmp_file in [
    tmp_fast_soc,
    tmp_slow_soc,
    tmp_v_fast_trimmed,
    tmp_v_slow_trimmed,
    tmp_precip_trimmed,
    tmp_check_dams
]:
    try:
        os.remove(tmp_file)
    except OSError:
        pass

print("Temporary memmap files removed. All done!")
