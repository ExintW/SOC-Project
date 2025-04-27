import sys
import os
import time
import contextlib
import io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from globals import *  
from htgy.D_Prediction_Model.htgy_SOC_model_with_river_basin import run_model
from Validate_RUSLE_Factors import run_valid

def suppress_print(func, *args, **kwargs):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        return func(*args, **kwargs)

def grid_search(a_range, b_range, c_range):
    best_a = a_range[0]
    best_b = b_range[0]
    best_c = c_range[0]
    best_rmse = 1e6
    
    max_err_inc_count = 2   # for early stopping
    
    for a in a_range:
        for b in b_range:
            err_inc_counter = 0
            before_rmse = 1e6   # for early stopping
            for c in c_range:
                print(f"\n#######################################################################")
                print(f"Running a={a}, b={b}, c={c}...")
                print(f"Cur best RMSE = {best_rmse}, a={best_a}, b={best_b}, c={best_c}")
                print(f"#######################################################################\n")
                
                # NOTE for lwk: if the following two lines doesn't work, replace with:
                # run_model(a, b, c)
                # cur_rmse = run_valid()
                suppress_print(run_model, a=a, b=b, c=c, start_year=2007, end_year=2018, past_year=1992, future_year=None)
                cur_rmse = suppress_print(run_valid)
                
                if cur_rmse < best_rmse:
                    best_a = a
                    best_b = b
                    best_c = c
                    best_rmse = cur_rmse
                elif before_rmse < cur_rmse:
                    err_inc_counter += 1
                before_rmse = cur_rmse
                
                if err_inc_counter >= max_err_inc_count:
                    break
                
    return best_a, best_b, best_c, best_rmse
    

if __name__ == "__main__":
    a_range = [-1.9]
    b_range = [1.80, 1.85, 1.90, 1.95, 2.0]
    c_range = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    total_param_sets = len(a_range) * len(b_range) * len(c_range)
    
    print(f"\n#######################################################################")
    print(f"Number of a: {len(a_range)}, b: {len(b_range)}, c:{len(c_range)}")
    print(f"Total running: {total_param_sets} param sets")
    print(f"Theoretical max runtime = {(1000 * total_param_sets / 60 / 60):.2f} hrs")
    print(f"#######################################################################\n")
    
    start_time = time.time()
    best_a, best_b, best_c, best_rmse = grid_search(a_range, b_range, c_range)
    end_time = time.time()
    
    print(f"\n#######################################################################")
    print("Grid search finished!")
    print(f"Grid search took {end_time - start_time / 3600} hrs")
    print(f"best_a: {best_a}, best_b: {best_b}, best_c: {best_c}, best_rmse: {best_rmse}")
    print(f"#######################################################################\n")
    
    with open(OUTPUT_DIR / 'grid_search_result.txt', 'w') as f:
        f.write(f"Grid search took {end_time - start_time / 3600} hrs\n")
        f.write(f"best_a: {best_a}, best_b: {best_b}, best_c: {best_c}, best_rmse: {best_rmse}\n")