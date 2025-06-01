# ============================================================
# Script to combine .npz files spanning different time ranges
# into unified dynamic_data.npz and static_data.npz files,
# using memory-mapped arrays and trimming v_* and precip to
# the 2007–2100 window.
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

# 1) IMPORTS
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
v_fast_paths = [
    PROCESSED_DIR / "V_fast_1950-2000.npz",
    PROCESSED_DIR / "V_fast_2001-2014.npz",
    PROCESSED_DIR / "V_fast_245_2015-2100.npz"
]
v_slow_paths = [
    PROCESSED_DIR / "V_slow_1950-2000.npz",
    PROCESSED_DIR / "V_slow_2001-2014.npz",
    PROCESSED_DIR / "V_slow_245_2015-2100.npz"
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
                   from each .npz; otherwise, uses the first key in each archive.

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
        arr = archive[key]  # memmap read-only
        T_i, H_i, W_i = arr.shape

        if dtype is None:
            dtype = arr.dtype
            height, width = H_i, W_i
        else:
            if (H_i, W_i) != (height, width):
                raise ValueError(
                    f"Shape mismatch: {p} has shape {arr.shape}, expected [*, {height}, {width}]."
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
        src_arr = archive[key]  # memmap slice
        combined_memmap[cursor: cursor + T_i, :, :] = src_arr
        cursor += T_i
        archive.close()

    return combined_memmap

# 4) COMBINE & TRIM DYNAMIC ARRAYS + PRINT SHAPES
print("Combining dynamic arrays (pre-trim) into memory‐mapped files...")

# 4a) soc_fast (2007–2024 + 2025–2100) → already exactly 2007–2100
soc_fast_memmap = combine_npz_to_memmap(
    source_paths       = fast_soc_paths,
    output_memmap_path = tmp_fast_soc
)
print("soc_fast combined shape (1128×844×1263):", soc_fast_memmap.shape)

# 4b) soc_slow (2007–2024 + 2025–2100) → already exactly 2007–2100
soc_slow_memmap = combine_npz_to_memmap(
    source_paths       = slow_soc_paths,
    output_memmap_path = tmp_slow_soc
)
print("soc_slow combined shape (1128×844×1263):", soc_slow_memmap.shape)

# 4c) v_fast (1950–2000 + 2001–2014 + 2015–2100) → trim to 2007–2100
v_fast_2001 = np.load(v_fast_paths[1], mmap_mode="r")
v_fast_2015 = np.load(v_fast_paths[2], mmap_mode="r")
key_fast_2001 = v_fast_2001.files[0]
key_fast_2015 = v_fast_2015.files[0]
arr2001 = v_fast_2001[key_fast_2001]  # shape (168, 844, 1263)
arr2015 = v_fast_2015[key_fast_2015]  # shape (1032, 844, 1263)

# Compute trimmed memmap for v_fast (1128 months)
months_2007_2100 = (2100 - 2007 + 1) * 12  # = 94 years * 12 = 1128
H, W = arr2001.shape[1], arr2001.shape[2]
v_fast_trimmed = np.memmap(
    tmp_v_fast_trimmed,
    dtype=arr2001.dtype,
    mode="w+",
    shape=(months_2007_2100, H, W)
)
# Fill Jan2007–Dec2014 (96 months)
start_idx_2007_2001 = (2007 - 2001) * 12  # = 6*12 = 72
v_fast_trimmed[0: (168 - start_idx_2007_2001), :, :] = arr2001[start_idx_2007_2001:168, :, :]
# Fill Jan2015–Dec2100 (1032 months)
v_fast_trimmed[(168 - start_idx_2007_2001):, :, :] = arr2015[0:1032, :, :]
print("v_fast trimmed shape (1128×844×1263):", v_fast_trimmed.shape)
v_fast_2001.close()
v_fast_2015.close()

# 4d) v_slow (1950–2000 + 2001–2014 + 2015–2100) → trim to 2007–2100
v_slow_2001 = np.load(v_slow_paths[1], mmap_mode="r")
v_slow_2015 = np.load(v_slow_paths[2], mmap_mode="r")
key_slow_2001 = v_slow_2001.files[0]
key_slow_2015 = v_slow_2015.files[0]
arr2001_slow = v_slow_2001[key_slow_2001]  # (168, 844, 1263)
arr2015_slow = v_slow_2015[key_slow_2015]  # (1032, 844, 1263)

v_slow_trimmed = np.memmap(
    tmp_v_slow_trimmed,
    dtype=arr2001_slow.dtype,
    mode="w+",
    shape=(months_2007_2100, H, W)
)
# Fill Jan2007–Dec2014 (96 months)
v_slow_trimmed[0: (168 - start_idx_2007_2001), :, :] = arr2001_slow[start_idx_2007_2001:168, :, :]
# Fill Jan2015–Dec2100 (1032 months)
v_slow_trimmed[(168 - start_idx_2007_2001):, :, :] = arr2015_slow[0:1032, :, :]
print("v_slow trimmed shape (1128×844×1263):", v_slow_trimmed.shape)
v_slow_2001.close()
v_slow_2015.close()

# 4e) precip (tp_1950–2024 + PR_245_2015–2100) → trim to 2007–2100
tp = np.load(precip_paths[0], mmap_mode="r")
pr = np.load(precip_paths[1], mmap_mode="r")
key_tp = tp.files[0]
key_pr = pr.files[0]
arr_tp = tp[key_tp]  # shape (900, 844, 1263)
arr_pr = pr[key_pr]  # shape (1032, 844, 1263)

# Create memmap for precip 2007–2100 (1128 months)
precip_trimmed = np.memmap(
    tmp_precip_trimmed,
    dtype=arr_tp.dtype,
    mode="w+",
    shape=(months_2007_2100, H, W)
)
# Fill Jan2007–Dec2024 (216 months)
start_idx_2007_tp = (2007 - 1950) * 12  # = 57*12 = 684
end_idx_2024_tp   = 75 * 12            # = 900
precip_trimmed[0: (end_idx_2024_tp - start_idx_2007_tp), :, :] = arr_tp[start_idx_2007_tp:end_idx_2024_tp, :, :]
# Fill Jan2025–Dec2100 (912 months)
start_idx_2025_pr = (2025 - 2015) * 12  # = 10*12 = 120
precip_trimmed[(end_idx_2024_tp - start_idx_2007_tp):, :, :] = arr_pr[start_idx_2025_pr:(start_idx_2025_pr + (2100 - 2025 + 1) * 12), :, :]
print("precip trimmed shape (1128×844×1263):", precip_trimmed.shape)
tp.close()
pr.close()

# 4f) check_dams (2007–2024 + 2025–2100) → already exactly 2007–2100
check_dams_memmap = combine_npz_to_memmap(
    source_paths       = dam_paths,
    output_memmap_path = tmp_check_dams
)
print("check_dams combined shape (1128×844×1263):", check_dams_memmap.shape)

print("\nFinished combining & trimming dynamic arrays.\n")

# 5) LOAD STATIC MASKS
print("Loading static masks...")

# 5a) Load DEM
dem_archive = np.load(dem_path)
dem_key = dem_archive.files[0]
dem = dem_archive[dem_key]  # 2D array [844, 1263]
dem_archive.close()
print("dem shape:", dem.shape)

# 5b) Load masks from precomputed_masks.npz
mask_archive = np.load(mask_path)
loess_border_mask      = mask_archive["loess_border_mask"]      # bool [844, 1263]
river_mask             = mask_archive["river_mask"]             # bool [844, 1263]
small_boundary_mask    = mask_archive["small_boundary_mask"]    # int/bool [844, 1263]
large_boundary_mask    = mask_archive["large_boundary_mask"]    # int/bool [844, 1263]
small_outlet_mask      = mask_archive["small_outlet_mask"]      # bool [844, 1263]
large_outlet_mask      = mask_archive["large_outlet_mask"]      # bool [844, 1263]
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

# 6a) dynamic_data.npz
np.savez(
    OUTPUT_DIR / "dynamic_data.npz",
    soc_fast   = soc_fast_memmap,
    soc_slow   = soc_slow_memmap,
    v_fast     = v_fast_trimmed,
    v_slow     = v_slow_trimmed,
    precip     = precip_trimmed,
    check_dams = check_dams_memmap
)

# 6b) static_data.npz (7 arrays)
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
