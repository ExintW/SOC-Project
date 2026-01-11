from config import *

class INIT_VALUES:
    SOC = None
    SOC_valid = None
    DEM = None
    SAND = None
    SILT = None
    CLAY = None
    LANDUSE = None
    REGION = None
    SLOPE = None
    K_fast = None
    K_slow = None
    C_fast = None
    C_slow = None
    SOC_PAST_FAST = None
    SOC_PAST_SLOW = None
    UNet_Model = None
    LAI_PAST = []
    LS_FACTOR = None
    P_FACTOR = None
    SORTED_INDICES = None
    
    @classmethod
    def reset(cls):
        for key in list(cls.__dict__):
            if not key.startswith('__') and not callable(getattr(cls, key)):
                setattr(cls, key, None)

class MAP_STATS:
    df_dam = None
    df_prop = None
    
    grid_x = None
    grid_y = None

    border_geom = None
    
    p_fast_grid = None
    
    large_outlet_mask = None
    small_outlet_mask = None
    
    small_boundary_mask = None
    large_boundary_mask = None

    border_mask = None
    river_mask = None
    
    C_fast_current = None
    C_slow_current = None
    
    C_fast_prev = None
    C_slow_prev = None

    C_fast_equil_list = []
    C_slow_equil_list = []
    
    total_C_matrix = None
    dam_rem_cap_matrix = None
    
    dam_cur_stored = None

    low_mask = None
    Low_Point_Capacity = None
    Low_Point_DEM_Dif = None

    REG_counter = REG_FREQ
    
    C_total_Past_Valid_list = []

    @classmethod
    def reset(cls):
        for key in list(cls.__dict__):
            if not key.startswith('__') and not callable(getattr(cls, key)):
                setattr(cls, key, None)