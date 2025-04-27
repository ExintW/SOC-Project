import sys
import os
import time
import contextlib
import io
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from globals import *  
from htgy.D_Prediction_Model.htgy_SOC_model_with_river_basin import run_model

def suppress_print(func, *args, **kwargs):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        return func(*args, **kwargs)

def grid_search_init(fraction_range, a, b, c, start_year, past_year):
    best_frac = fraction_range[0]
    best_rmse = 1e6
    
    for frac in fraction_range:
        print(f"\n#######################################################################")
        print(f"Running frac={frac}...")
        print(f"Cur best RMSE = {best_rmse}, frac={best_frac}")
        print(f"#######################################################################\n")

        # rmse = suppress_print(run_model, a=a, b=b, c=c, start_year=start_year, end_year=None, past_year=past_year, future_year=None)
        rmse = run_model(a=a, b=b, c=c, start_year=start_year, end_year=None, past_year=past_year, future_year=None, fraction=frac)
        if rmse < best_rmse:
            best_frac = frac
            best_rmse = rmse
    
    return best_frac, best_rmse
        
if __name__ == "__main__":
    start_value = 0.95
    end_value = 0.7
    step_size = -0.015
    
    a = -1.9
    b = 1.8
    c = 3
    
    fraction_range = np.arange(start_value, end_value - 1e-8, step_size)
    start_year = 2007   # year of init condition
    past_year = 1980
    
    
    total_param_sets = len(fraction_range)
    
    print(f"\n#######################################################################")
    print(f"Total running: {total_param_sets} param sets")
    print(f"Theoretical max runtime = {((start_year - past_year)*38.46 * total_param_sets / 60 / 60):.2f} hrs")
    print(f"#######################################################################\n")
    
    start_time = time.time()
    best_frac, best_rmse = grid_search_init(fraction_range, a, b, c, start_year, past_year)
    end_time = time.time()
    
    print(f"\n#######################################################################")
    print("Grid search finished!")
    print(f"Grid search took {end_time - start_time / 3600} hrs")
    print(f"best_frac: {best_frac}, best_rmse: {best_rmse}")
    print(f"#######################################################################\n")
    
    with open(OUTPUT_DIR / 'Past_SOC_grid_search_result.txt', 'w') as f:
        f.write(f"Grid search took {end_time - start_time / 3600} hrs\n")
        f.write(f"best_frac: {best_frac}, best_rmse: {best_rmse}\n")