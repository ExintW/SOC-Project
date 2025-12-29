import os
import sys
import numpy as np
import scipy.ndimage as ndimage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import Paths 
from global_structs import MAP_STATS

class TeeOutput:
    def __init__(self, file, original_stdout):
        self.file = file
        self.original_stdout = original_stdout
    
    def write(self, text):
        self.file.write(text)
        if ("Processing year" in text or 
            "=======================================================================" in text or
            "Year" in text and "Month" in text or
            "Completed simulation for Year" in text):
            self.original_stdout.write(text)
        self.file.flush()
    
    def flush(self):
        self.file.flush()
        self.original_stdout.flush()

def gaussian_blur_with_nan(data, sigma=1):
    mask = ~np.isnan(data)
    data_filled = np.where(mask, data, 0)

    blurred_data = ndimage.gaussian_filter(data_filled, sigma=sigma)
    blurred_mask = ndimage.gaussian_filter(mask.astype(float), sigma=sigma)

    with np.errstate(invalid='ignore'):
        result = blurred_data / blurred_mask
    result[~MAP_STATS.border_mask] = np.nan
    return result