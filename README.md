# SOC-Project — Soil Organic Carbon Dynamics on the Loess Plateau

A spatial process model of **Soil Organic Carbon (SOC)** dynamics on the Loess
Plateau, China (`htgy` = 黄土高原). On a ~1 km monthly grid it couples:

- **RUSLE** soil erosion (`A = R · K · LS · C · P`),
- a **two-pool** (fast / slow) SOC reaction model with humification,
- slope-driven **sediment & SOC transport** routing (cell-to-cell, high → low elevation),
- **check-dam** interception and storage, and
- **river export** of out-of-basin SOC.

The model runs in three directions: **present** (historical forcing), **future**
(CMIP6 scenarios), and **past** (reverse-in-time reconstruction of historical SOC,
regularized toward known 1980 SOC).

> For the end-to-end map of raw inputs → processing scripts → model → outputs, see
> **[`README_SOC_Data_Flow.md`](README_SOC_Data_Flow.md)**.
> For architecture and conventions aimed at contributors, see **[`CLAUDE.md`](CLAUDE.md)**.

## Requirements

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- The model reads large geospatial inputs from `Raw_Data/` and `Processed_Data/`
  (DEM, shapefiles, ERA5/CMIP6 NetCDF, 1980 SOC). These are not all reproducible
  locally — ERA5 requires a `cdsapi` account, CMIP6 and SOM are external downloads.

## Setup

```bash
uv sync          # creates an isolated venv and installs pinned dependencies from uv.lock
```

This installs the full stack (numpy, pandas, xarray, geopandas, rasterio, rioxarray,
shapely, scikit-learn, scipy, torch, whitebox, opencv, netCDF4, cdsapi, …).

## Running the model

The entry point is `htgy/D_Prediction_Model/main.py`. It takes **no command-line
arguments** — all behavior is configured by editing
`htgy/D_Prediction_Model/config.py` first.

```bash
uv run python htgy/D_Prediction_Model/main.py
```

> Note: the `main.py` at the repository root is an unused placeholder. Use the path above.

### Configuring a run

Open `htgy/D_Prediction_Model/config.py` and set the simulation bounds and options.
Set an unused bound to `None` to disable that phase. Key parameters:

| Parameter        | Meaning                                                              |
| ---------------- | ------------------------------------------------------------------- |
| `INIT_YEAR`      | Year of the initial SOC state (start of present run)                |
| `END_YEAR`       | Last year of the present forward run (`None` to skip)               |
| `FUTURE_YEAR`    | Last year of the CMIP6 future run; starts at `END_YEAR + 1` (`None` to skip) |
| `PAST_YEAR`      | Earliest year of the reverse (past) run (`None` to skip)            |
| `EQUIL_YEAR`     | Forward-computed year used as the equilibrium seed for the past run |
| `RUN_FROM_EQUIL` | If `True`, the past run starts from the equilibrium state           |
| `USE_TIKHONOV`   | Enable L2 regularization toward a prior in the reverse run          |
| `PAST_KNOWN`     | Year of the one known historical SOC field (default 1980) used as the reverse-run prior |
| `SAVE_NC`        | Also export results as NetCDF                                       |
| `USE_PARQUET`    | Store per-timestep output as Parquet (else CSV)                     |
| `CLEAN_OUTDIR`   | Wipe previous `Output/Data` and `Output/Figure` contents before running |

The full parameter set (vegetation scaling, pool caps, humification `ALPHA`,
Gaussian-blur prior, spatial regularization, etc.) is documented inline in
`config.py`, and the exact values used are written to `Output/out.log` on every run.

### First run

On the first run the model builds two caches (slow the first time, reused afterward):

- `Processed_Data/LS_factor.npy` — RUSLE LS factor, computed from the DEM via WhiteboxTools
- `Processed_Data/precomputed_masks.npz` — basin / river / border rasterized masks

**Delete these caches** if you change the DEM, the region border, or the basin
shapefiles, since they are otherwise silently reused.

## Outputs

Written to `Output/` (see `README_SOC_Data_Flow.md` for the full list):

- `Output/Data/SOC_terms_<year>_<mm>_River.parquet` — per-timestep grids of every SOC flux term
- `Output/Figure/SOC_<year>_<mm>.png` — per-timestep maps
- `Output/SOC_<start>_<end>.mp4` — animation of the run
- `Output/Total_C_<start>-<end>_monthly.nc`, `Dam_rem_Cap_*.nc` — NetCDF time series (when `SAVE_NC`)
- `Output/out.log` — run log plus the full parameter snapshot

## Repository layout

```
htgy/
  A_Data_visualization/   Plotting and animation (png → mp4/gif)
  B_Data_processing/      Resample raw ERA5 / CMIP6 / SOC / DEM into Processed_Data/
  C_Regression/           Empirical fits (e.g. LAI → vegetation input)
  D_Prediction_Model/     The SOC model (main entry point)
  E_Validation/           Validation against observations, parameter grid searches
  Global_Model/           Variant of the model configured for a larger/global region
  Legacy_Model/           Deprecated earlier implementation (do not build on it)
  paths.py, globals.py    Centralized Raw_Data / Processed_Data / Output path resolution

Raw_Data/         External inputs (DEM, shapefiles, ERA5, CMIP6, 1980 SOC)
Processed_Data/   Model-ready inputs produced by B_Data_processing/ (+ cached factors/masks)
Output/           Model results, figures, logs
```

Scripts in `A`–`C` and `E` are run individually (e.g.
`uv run python htgy/B_Data_processing/Resample_ERA5.py`) and are not orchestrated by
the model entry point.

## Testing

There is currently no automated test suite. Validation is performed via the scripts
in `htgy/E_Validation/`, which compare model output against observed SOC and erosion
data and report metrics such as RMSE.
