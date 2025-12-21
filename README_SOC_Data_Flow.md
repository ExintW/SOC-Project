# SOC Model Data Flow Diagram

This document illustrates the complete data processing pipeline of the SOC (Soil Organic Carbon) model, from raw data inputs to final outputs.

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAW DATA INPUT                                   │
└─────────────────────────────────────────────────────────────────────────────┘

 Raw_Data/Loess_Plateau_Points.csv ──────────────┐
 Raw_Data/Loess_Plateau_vector_border.shp ───────┤
 Raw_Data/htgyDEM.tif ────────────────────────────┤
 Raw_Data/骨干坝/骨干坝.shp ───────────────────────┤
 Raw_Data/中型坝/中型坝.shp ───────────────────────┤
 Raw_Data/k1_halfDegree.tif ──────────────────────┤
 Raw_Data/k2_halfDegree.tif ──────────────────────┤
                                                  ↓
                  htgy/B_Data_processing/generate_processed_dataset.py
                                                  ↓
         Processed_Data/Resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv
         Processed_Data/htgy_Dam_with_matched_points.csv

────────────────────────────────────────────────────────────────────────────────

 Raw_Data/1980_SOM/SOM.nc ───────────────────────┐
                                                  ↓
                  htgy/B_Data_processing/Resample_1980_SOC.py
                                                  ↓
                  Processed_Data/soc_resampled_1980_matrix.npz

────────────────────────────────────────────────────────────────────────────────

 Raw_Data/CMIP6/lai_Lmon_BCC-CSM2-HR_hist-1950_r1i1p1f1_gn_195001-200012.nc ──┐
 Raw_Data/CMIP6/lai_Lmon_BCC-CSM2-HR_hist-1950_r1i1p1f1_gn_200101-201412.nc ──┤
 Raw_Data/CMIP6/lai_Lmon_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_201501-210012.nc ─────┤
 Raw_Data/CMIP6/pr_Amon_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_201501-210012.nc ──────┤
 Raw_Data/CMIP6/lai_Lmon_BCC-CSM2-MR_ssp245_r1i1p1f1_gn_201501-210012.nc ─────┤
 Raw_Data/CMIP6/pr_Amon_BCC-CSM2-MR_ssp245_r1i1p1f1_gn_201501-210012.nc ──────┤
 Raw_Data/CMIP6/lai_Lmon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_201501-210012.nc ─────┤
 Raw_Data/CMIP6/pr_Amon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_201501-210012.nc ──────┤
                                                  ↓
                  htgy/B_Data_processing/Generate_All_CMIP6_Data.py
                                                  ↓
         Processed_Data/CMIP6_Data_Monthly_Resampled/resampled_lai_points_1950-2000.nc
         Processed_Data/CMIP6_Data_Monthly_Resampled/resampled_lai_points_2001-2014.nc
         Processed_Data/CMIP6_Data_Monthly_Resampled/resampled_lai_points_2015-2100_245.nc
         Processed_Data/CMIP6_Data_Monthly_Resampled/resampled_lai_points_2015-2100_585.nc
         Processed_Data/CMIP6_Data_Monthly_Resampled/resampled_pr_points_2015-2100_245.nc
         Processed_Data/CMIP6_Data_Monthly_Resampled/resampled_pr_points_2015-2100_585.nc
         Processed_Data/CMIP6_Data_Monthly_Resampled/resampled_lai_2015-2100_126.nc
         Processed_Data/CMIP6_Data_Monthly_Resampled/resampled_lai_2015-2100_245.nc
         Processed_Data/CMIP6_Data_Monthly_Resampled/resampled_lai_2015-2100_585.nc
         Processed_Data/CMIP6_Data_Monthly_Resampled/resampled_pr_2015-2100_126.nc
         Processed_Data/CMIP6_Data_Monthly_Resampled/resampled_pr_2015-2100_245.nc
         Processed_Data/CMIP6_Data_Monthly_Resampled/resampled_pr_2015-2100_585.nc
         Processed_Data/CMIP6_Data_Monthly_Resampled/annual_LAI_stats_1950-2000.csv
         Processed_Data/CMIP6_Data_Monthly_Resampled/annual_LAI_stats_2000-2015.csv
         Processed_Data/CMIP6_Data_Monthly_Resampled/annual_LAI_stats_1950-2015.csv

────────────────────────────────────────────────────────────────────────────────

 Raw_Data/ERA5/1950.nc, 1951.nc, ..., 2025.nc ──┐
                                                  ↓
                  htgy/B_Data_processing/Resample_ERA5.py
                                                  ↓
         Processed_Data/ERA5_Data_Monthly_Resampled/resampled_1950.nc
         Processed_Data/ERA5_Data_Monthly_Resampled/resampled_1951.nc
         ...
         Processed_Data/ERA5_Data_Monthly_Resampled/resampled_2025.nc

────────────────────────────────────────────────────────────────────────────────

 Raw_Data/htgyDEM.tif ────────────────────────────┐
 Processed_Data/Resampled_Loess_Plateau_1km_with_DEM_region_k1k2_labeled.csv ─┤
                                                  ↓
                  htgy/D_Prediction_Model/low_point_utils.py
                                                  ↓
                  Processed_Data/Low_Point_Summary.csv

════════════════════════════════════════════════════════════════════════════════
                              MAIN MODEL EXECUTION
════════════════════════════════════════════════════════════════════════════════

                  htgy/D_Prediction_Model/htgy_SOC_model_with_river_basin.py

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ [AUTO-GENERATED ON FIRST RUN]                                           │
  │                                                                         │
  │ IF Processed_Data/LS_factor.npy NOT EXISTS:                            │
  │    → Call functions from htgy/D_Prediction_Model/RUSLE_Calculations.py │
  │    → Use: Raw_Data/htgyDEM.tif                                         │
  │    → Generate: Processed_Data/LS_factor.npy                            │
  │    → Time: ~5-10 minutes                                               │
  │                                                                         │
  │ IF Processed_Data/precomputed_masks.npz NOT EXISTS:                    │
  │    → Call functions from htgy/D_Prediction_Model/River_Basin.py        │
  │    → Use: Raw_Data/Loess_Plateau_vector_border.shp                     │
  │           Raw_Data/River_Basin/htgy_River_Basin.shp                    │
  │           Raw_Data/River_Basin/94_area.shp                             │
  │           Raw_Data/China_River/ChinaRiver_main.shp                     │
  │    → Generate: Processed_Data/precomputed_masks.npz                    │
  │    → Time: ~2-5 minutes                                                │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ [INITIALIZATION]                                                        │
  │                                                                         │
  │  • Load all processed data                                             │
  │  • Partition SOC into fast/slow pools                                  │
  │  • Load dam capacity and construction year                             │
  │  • Load RUSLE factors (LS, K)                                          │
  │  • Load river basin masks                                              │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ [MONTHLY SIMULATION LOOP]                                               │
  │                                                                         │
  │  For each month:                                                        │
  │    1. Load monthly LAI and precipitation from:                         │
  │       • Processed_Data/ERA5_Data_Monthly_Resampled/*.nc (historical)   │
  │       • Processed_Data/CMIP6_Data_Monthly_Resampled/*.nc (future)      │
  │    2. Calculate RUSLE erosion: A = R × K × LS × C × P                  │
  │    3. SOC transport (Numba-accelerated flow routing)                   │
  │    4. Low point deposition                                             │
  │    5. Dam interception and storage update                              │
  │    6. River removal (out-of-basin SOC)                                 │
  │    7. Update SOC pools: C(t+1) = C(t) + ΔC                             │
  │    8. Save monthly snapshot                                            │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ [POST-PROCESSING]                                                       │
  │                                                                         │
  │  • Export NetCDF time series                                           │
  │  • Export NPZ arrays (optional)                                        │
  │  • Generate visualization figures                                      │
  │  • Create MP4 animation                                                │
  │  • Calculate validation metrics (RMSE)                                 │
  └─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                             FINAL OUTPUTS                                   │
└─────────────────────────────────────────────────────────────────────────────┘

  Output/Total_C_{start_year}-{end_year}_monthly.nc
  Output/Dam_rem_Cap_{start_year}-{end_year}_monthly.nc
  Output/Fast_SOC_year_{start_year}-{end_year}.npz (optional)
  Output/Slow_SOC_year_{start_year}-{end_year}.npz (optional)
  Output/Active_dams_{start_year}-{end_year}.npz (optional)
  Output/DEM.npz (optional)
  Output/Data/SOC_{scenario}_{idx}/SOC_terms_{year}_{month}_{suffix}.parquet
  Output/Figure/SOC_initial.png
  Output/Figure/SOC_{year}_{month}.png
  Output/SOC_{start_year}_{end_year}.mp4
  Output/out.log
  Output/SOC_1980_Total_cleaned.npz (validation)
  Output/SOC_1980_Total_predicted.npz (validation)

```

---

**Document Version**: v2.0  
**Last Updated**: 2025-11-22  
**Maintainer**: SOC-Project Team
