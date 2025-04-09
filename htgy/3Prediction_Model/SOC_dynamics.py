from globalss import *
from globals import *  

# =============================================================================
# VEGETATION INPUT & UPDATED SOC DYNAMIC MODEL
# =============================================================================

def vegetation_input(LAI):
    """
    Compute vegetation input based on LAI using an empirical formula.
    E.g., V = a * LAI + b
    """
    return 0.08760361 * LAI - 0.00058271

def soc_dynamic_model(C_fast, C_slow,
                      soc_loss_g_kg_month, D_soil, D_soc, V,
                      K_fast, K_slow, p_fast_grid, dt, M_soil, lost_soc):
    """
    Update SOC pools (g/kg) for one month.
    - Erosion removes SOC (soc_loss_g_kg_month).
    - Deposition adds SOC (converted from D_soc to g/kg).
    - Vegetation adds new SOC input.
    - Reaction (decay) reduces each pool at rates K_fast, K_slow.
    - Lost SOC (e.g., to rivers) is subtracted.
    """
    # Erosion partitioned into fast & slow
    erosion_fast = -soc_loss_g_kg_month * p_fast_grid
    erosion_slow = -soc_loss_g_kg_month * (1 - p_fast_grid)

    # Deposition: (D_soc * 1000) / M_soil -> convert t -> g, then per kg soil
    deposition_concentration = (D_soc * 1000.0) / M_soil
    deposition_fast = deposition_concentration * p_fast_grid
    deposition_slow = deposition_concentration * (1 - p_fast_grid)

    # Vegetation input
    vegetation_fast = V * p_fast_grid
    vegetation_slow = V * (1 - p_fast_grid)

    # Reaction/decay
    reaction_fast = -K_fast * C_fast
    reaction_slow = -K_slow * C_slow

    # Lost SOC partition
    lost_fast = lost_soc * p_fast_grid
    lost_slow = lost_soc * (1 - p_fast_grid)

    # Update each pool
    C_fast_new = np.maximum(
        C_fast + (erosion_fast + deposition_fast + vegetation_fast + reaction_fast - lost_fast) * dt,
        0
    )
    C_slow_new = np.maximum(
        C_slow + (erosion_slow + deposition_slow + vegetation_slow + reaction_slow - lost_slow) * dt,
        0
    )
    return C_fast_new, C_slow_new