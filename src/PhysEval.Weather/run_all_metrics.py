#!/usr/bin/env python
"""
Physics Evaluation Runner — WeatherBench 2 Zarr Streaming

Streams Model predictions and reference ground-truth data from public WB2
Zarr buckets, computes all physics metrics at multiple forecast horizons
(6 h, Day 5, Day 10), and saves a single long-format CSV.

Output Format (melted long)
---------------------------
  date | lead_time_hours | metric_name | model_value | ref_value | n_levels | sp_method

Data Sources
------------
  Aurora : gs://weatherbench2/datasets/aurora/2022-1440x721.zarr
  ERA5   : gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr

Usage
-----
  python run_physics_evaluation.py --year 2022
  python run_physics_evaluation.py --year 2022 --workers 16
  python run_physics_evaluation.py --dates 2022-01-01 2022-01-02
"""

from __future__ import annotations

import argparse
import calendar
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

# Companion library (same directory)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from physics_metrics import (
    compute_conservation_scalars,
    compute_drift_percentages,
    compute_drift_slope,
    compute_pure_tcwv,
    _find_effective_resolution,
    compute_geostrophic_imbalance,
    compute_hydrostatic_imbalance,
    compute_ke_spectrum,
    compute_q_spectrum,
    compute_spectral_scores,
    derive_surface_pressure,
    get_grid_cell_area,
    _find_var,
    _detect_level_dim,
    _detect_pred_td_dim,
    SP_NAMES,
    MSL_NAMES,
    T_NAMES,
    T2M_NAMES,
    ZSFC_NAMES,
    Q_NAMES,
    U_NAMES,
    V_NAMES,
    PHI_NAMES,
)

# Suppress xarray FutureWarning about timedelta decoding
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*prediction_timedelta.*")


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL_ZARR = "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr"
REF_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

# IFS HRES t=0 (analysis) as alternative reference
IFS_T0_ZARR = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
IFS_T0_LOWRES_ZARR = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-512x256_equiangular_conservative.zarr"

OUTPUT_DIR = Path.home() / "aurora_thesis" / "thesis" / "benchmark" / "results"

# Target lead times: (label, timedelta)
# Using 12h as the first lead time to align with NeuralGCM (which lacks 6h step)
LEAD_TIMES: list[tuple[str, np.timedelta64]] = [
    ("12h",  np.timedelta64(12,  "h")),
    ("5d",   np.timedelta64(120, "h")),
    ("10d",  np.timedelta64(240, "h")),
]

# Map each target lead_td to the END of the drift-regression window.
# Start is always 12 h (to align with NeuralGCM).
DRIFT_WINDOW_END: dict[int, np.timedelta64] = {
    12:  np.timedelta64(24,  "h"),   # 12h target → window 12h–24h
    120: np.timedelta64(120, "h"),   # 5d  target → window 12h–120h
    240: np.timedelta64(240, "h"),   # 10d target → window 12h–240h
}

DEFAULT_WORKERS = 16


# ============================================================================
# Zarr I/O
# ============================================================================

def open_zarr_anonymous(url: str) -> xr.Dataset:
    """Open a public GCS Zarr store without authentication."""
    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    # Sanitise variable AND dimension names (some Zarrs have trailing whitespace)
    rename = {}
    for v in ds.data_vars:
        if v != v.strip():
            rename[v] = v.strip()
    for d in ds.dims:
        if d != d.strip():
            rename[d] = d.strip()
    # Normalise short lat/lon dim names (e.g. GraphCast uses "lat"/"lon")
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def load_static_fields(ds_ref: xr.Dataset) -> xr.Dataset:
    """
    Extract static fields (z_sfc, land-sea mask) from Reference at time=0.
    """
    static_vars = {}

    def _extract_static(ds, name):
        var = ds[name]
        if "time" in var.dims:
            var = var.isel(time=0, drop=True)
        return var

    for name in ("geopotential_at_surface", "z_sfc", "orography"):
        if name in ds_ref.data_vars:
            static_vars[name] = _extract_static(ds_ref, name)
            break

    for name in ("land_sea_mask", "lsm"):
        if name in ds_ref.data_vars:
            static_vars[name] = _extract_static(ds_ref, name)
            break

    if not static_vars:
        raise ValueError(
            f"No static fields found. "
            f"Available: {list(ds_ref.data_vars)[:20]}"
        )

    return xr.Dataset(static_vars)


def _get_ps(
    ds: xr.Dataset,
    ds_static: xr.Dataset,
    level_dim: str = "level",
) -> xr.DataArray:
    """Return surface pressure, deriving from MSL if needed,
    with fallback to bottom-level barometric estimation."""
    # 1. Direct surface pressure variable
    sp_name = _find_var(ds, SP_NAMES)
    if sp_name is not None:
        sp = ds[sp_name]
        sp.attrs["derivation_method"] = "direct_sp"
        return sp

    # 2. Hypsometric from MSL (Standard Atmosphere)
    try:
        sp = derive_surface_pressure(ds, ds_static)
        sp.attrs["derivation_method"] = "hypsometric_msl_standard_atm"
        return sp
    except (ValueError, KeyError):
        pass

    raise ValueError(
        f"Cannot derive surface pressure: no SP variable and "
        f"hypsometric derivation from MSL failed. "
        f"Available vars: {list(ds.data_vars)}"
    )


# ============================================================================
# Grid Alignment
# ============================================================================


def _grids_match(
    ds_a: xr.Dataset,
    ds_b: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    atol: float = 1e-3,
) -> bool:
    """Return True if two datasets share the same lat/lon grid (order-insensitive)."""
    if ds_a.sizes.get(lat_name, 0) != ds_b.sizes.get(lat_name, 0):
        return False
    if ds_a.sizes.get(lon_name, 0) != ds_b.sizes.get(lon_name, 0):
        return False
    lat_a = np.sort(ds_a[lat_name].values)
    lat_b = np.sort(ds_b[lat_name].values)
    if not np.allclose(lat_a, lat_b, atol=atol):
        return False
    lon_a = np.sort(ds_a[lon_name].values)
    lon_b = np.sort(ds_b[lon_name].values)
    if not np.allclose(lon_a, lon_b, atol=atol):
        return False
    return True


def _align_ref_to_model(
    ds_ref: xr.Dataset,
    ds_model: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """
    Align Reference grid to match Model for spectral comparison.

    Only two cases are allowed:
    1. Exact shape match — grids are identical (or close enough).
    2. Reference has exactly 1 extra latitude row (e.g. 721 vs 720) — drop the
       pole row so the grids match.

    No interpolation is performed.  Any other mismatch raises an error.
    """
    n_ref = ds_ref.sizes.get(lat_name, 0)
    n_model = ds_model.sizes.get(lat_name, 0)
    n_lon_ref = ds_ref.sizes.get(lon_name, 0)
    n_lon_model = ds_model.sizes.get(lon_name, 0)

    if n_lon_ref != n_lon_model:
        raise ValueError(
            f"Longitude grid mismatch: Reference has {n_lon_ref}, "
            f"prediction has {n_lon_model}. Cannot align without interpolation."
        )

    if n_ref == n_model:
        result = ds_ref
    elif n_ref == n_model + 1:
        # Reference has 1 extra lat row (pole) — drop it
        lats = ds_ref[lat_name].values
        if lats[0] > lats[-1]:  # Descending (N→S): drop last row (south pole)
            result = ds_ref.isel({lat_name: slice(0, -1)})
        else:                   # Ascending (S→N): drop first row (south pole)
            result = ds_ref.isel({lat_name: slice(1, None)})
    else:
        raise ValueError(
            f"Latitude grid mismatch: Reference has {n_ref} rows, "
            f"prediction has {n_model}. Only exact match or 1-row "
            f"pole difference is supported (no interpolation)."
        )

    # Reassign latitude coordinates from prediction to ensure exact alignment
    result = result.assign_coords({lat_name: ds_model[lat_name].values})

    # Ensure (latitude, longitude) dimension order for spectral analysis
    if lat_name in result.dims and lon_name in result.dims:
        dims_list = list(result.dims)
        idx_lon = dims_list.index(lon_name)
        idx_lat = dims_list.index(lat_name)
        if idx_lon < idx_lat:
            dims_list[idx_lon], dims_list[idx_lat] = (
                dims_list[idx_lat], dims_list[idx_lon]
            )
            result = result.transpose(*dims_list)

    return result


# ============================================================================
# Date Resolution
# ============================================================================

def _resolve_dates(args) -> list[str]:
    """Build a list of ISO date strings from CLI arguments (00:00 init only)."""
    if args.dates:
        return [d if "T" in d else f"{d}T00:00:00" for d in args.dates]
    if args.month:
        year, month = args.month.split("-")
        n_days = calendar.monthrange(int(year), int(month))[1]
        return [f"{year}-{month}-{d:02d}T00:00:00" for d in range(1, n_days + 1)]
    year = args.year
    dates = []
    for m in range(1, 13):
        n_days = calendar.monthrange(year, m)[1]
        for d in range(1, n_days + 1):
            dates.append(f"{year}-{m:02d}-{d:02d}T00:00:00")
    return dates


# ============================================================================
# Single-Slice Evaluation → melted rows
# ============================================================================

def _evaluate_one(
    model_zarr_path: str,
    ref_zarr_path: str,
    date_str: str,
    lead_label: str,
    lead_td: np.timedelta64,
    counter: int,
    total: int,
    mode: str,
    verbose: bool,
    static_zarr_path: str | None = None,
    model_name: str = "model",
    extended_spectra: bool = False,
    sp_ablation: str = "default",
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Fetch, load, and evaluate one (date × lead_time) combination.
    Runs in a separate process, so it opens its own Zarr connections.

    Returns (summary_rows, ts_rows, spectrum_rows, lr_dist_rows)
    """
    # Prevent nested Dask multithreading inside the multiprocessing workers
    # which causes thread contention and slows down GCS I/O.
    dask.config.set(scheduler='synchronous')

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    _log(
        f"  [{counter}/{total}] "
        f"init={date_str}  lead={lead_label}  "
        f"— Connecting to dataset..."
    )

    # Open datasets inside the worker process to avoid pickling issues
    ds_model_full = None
    if mode in ("joint", "prediction", "model"):
        ds_model_full = open_zarr_anonymous(model_zarr_path)
    
    ds_ref_full = open_zarr_anonymous(ref_zarr_path)
    
    # Load static fields — from a dedicated static Zarr if provided
    # (needed when ref_zarr is e.g. HRES-T0 which lacks static fields)
    if static_zarr_path:
        ds_static_src = open_zarr_anonymous(static_zarr_path)
    else:
        ds_static_src = ds_ref_full
    ds_static = load_static_fields(ds_static_src)

    # Ensure static fields are in (latitude, longitude) order
    for var_name in list(ds_static.data_vars):
        v = ds_static[var_name]
        if "latitude" in v.dims and "longitude" in v.dims:
            if list(v.dims).index("longitude") < list(v.dims).index("latitude"):
                ds_static[var_name] = v.transpose("latitude", "longitude")

    z_sfc_name = _find_var(ds_static, ZSFC_NAMES)
    z_sfc = ds_static[z_sfc_name]
    
    # Grid cell area
    area = get_grid_cell_area(ds_ref_full.isel(time=0, drop=True))

    lead_hours = int(lead_td / np.timedelta64(1, "h"))
    # Cast to ns resolution so xarray .sel() matches the Reference Zarr time coord
    init_time = np.datetime64(date_str, "ns")
    valid_time = init_time + lead_td

    rows: list[dict] = []

    # Will be set after model data is loaded
    _n_levels: int | None = None
    _sp_method: str = "none"

    def _append(metric_name, model_val, ref_val=None):
        rows.append({
            "date": date_str,
            "lead_time_hours": lead_hours,
            "metric_name": metric_name,
            "model_value": model_val,
            "ref_value": ref_val,
            "n_levels": _n_levels,
            "sp_method": _sp_method,
        })

    _log(
        f"  [{counter}/{total}] "
        f"init={init_time}  lead={lead_label} ({lead_hours}h)  "
        f"valid={valid_time} — Data connected, processing..."
    )

    try:
        # ---- Fetch and load Model slice ----
        ds_model_t = None
        ps_model = None
        area_model = None
        _lead_td_mismatch = False

        if ds_model_full is not None:
            # Normalise dimension names (some zarrs have trailing whitespace)
            dim_rename = {d: d.strip() for d in ds_model_full.dims if d != d.strip()}
            if dim_rename:
                ds_model_full = ds_model_full.rename(dim_rename)

            ds_model_t = ds_model_full.sel(time=init_time)

            # Auto-detect prediction_timedelta dimension name
            pred_td_dim = _detect_pred_td_dim(ds_model_t)
            _lead_td_mismatch = False  # track if nearest != requested
            if pred_td_dim is not None and pred_td_dim in ds_model_t.dims:
                ds_model_t = ds_model_t.sel(
                    {pred_td_dim: lead_td}, method="nearest"
                )
                # Check if the actual selected timedelta matches requested
                actual_td = ds_model_t.coords.get(pred_td_dim)
                if actual_td is not None:
                    actual_td_val = actual_td.values
                    if isinstance(actual_td_val, np.timedelta64) and actual_td_val != lead_td:
                        _log(f"    [{counter}] ⚠ Requested lead={lead_td}, "
                             f"nearest available={actual_td_val} — skipping metrics for this lead time")
                        _lead_td_mismatch = True

            # Drop variables not needed for physics metrics to reduce I/O.
            # Critical for datasets with large chunks (e.g. FuXi chunks
            # span all 60 prediction timedeltas, so each var is ~120 MB).
            _NEEDED_VARS = set()
            my_sp_names = SP_NAMES
            my_msl_names = MSL_NAMES
            my_q_names = Q_NAMES
            
            if sp_ablation == "hypsometric":
                my_sp_names = ()
            elif sp_ablation == "ref_sp":
                my_sp_names = ()
                my_msl_names = ()
            elif sp_ablation == "dry_hydro":
                my_q_names = ()

            for names in (T_NAMES, PHI_NAMES, U_NAMES, V_NAMES,
                          my_q_names, my_msl_names, my_sp_names, T2M_NAMES, ZSFC_NAMES):
                _NEEDED_VARS.update(names)
            drop_vars = [v for v in ds_model_t.data_vars
                         if v.strip() not in _NEEDED_VARS]
            if drop_vars:
                _log(f"    [{counter}] Dropping {len(drop_vars)} unneeded vars: {drop_vars}")
                ds_model_t = ds_model_t.drop_vars(drop_vars)

            if "time" in ds_model_t.dims:
                 ds_model_t = ds_model_t.isel(time=0)

            # Ensure spatial dimensions are in standard (latitude, longitude) order
            if "latitude" in ds_model_t.dims and "longitude" in ds_model_t.dims:
                dims_list = list(ds_model_t.dims)
                idx_lon = dims_list.index("longitude")
                idx_lat = dims_list.index("latitude")
                if idx_lon < idx_lat:
                    _log(f"    [{counter}] Transposing spatial dims from (lon, lat) to (lat, lon)")
                    dims_list[idx_lon], dims_list[idx_lat] = dims_list[idx_lat], dims_list[idx_lon]
                    ds_model_t = ds_model_t.transpose(*dims_list)

            ds_model_t = ds_model_t.load()

            # ---- Static fields must match model grid shape ----
            # For static fields (orography, land-sea mask) that vary slowly,
            # interpolation to the model grid is acceptable when grids differ.
            lat_name = "latitude"
            lon_name = "longitude"
            model_grid_matches_ref = _grids_match(ds_model_t, ds_ref_full)

            ds_static_model = ds_static
            z_sfc_model = z_sfc
            if not model_grid_matches_ref:
                # Interpolate static fields to the model grid (nearest neighbour)
                interp_coords = {
                    lat_name: ds_model_t[lat_name].values,
                    lon_name: ds_model_t[lon_name].values,
                }
                ds_static_model = ds_static.interp(interp_coords, method="nearest")
                z_sfc_name = _find_var(ds_static_model, ZSFC_NAMES)
                if z_sfc_name is not None:
                    z_sfc_model = ds_static_model[z_sfc_name]
                _log(f"    [{counter}] Model grid differs from Reference — interpolating static fields")

            # Only derive surface pressure if humidity is available,
            # since ps is only used for mass/water/energy metrics (which
            # require q).  Skipping saves an expensive hypsometric calc.
            has_q = _find_var(ds_model_t, Q_NAMES) is not None
            # Auto-detect the level dimension name for this dataset
            model_level_dim = _detect_level_dim(ds_model_t)
            
            # Check if model has SP or MSL (needed to derive surface pressure independently)
            # Note: 3D geopotential alone is NOT sufficient - geopotential_interpolation
            # still requires Reference's surface geopotential (z_sfc), so models with only
            # 3D geopotential but no SP/MSL should use Reference sp directly.
            _has_sp = _find_var(ds_model_t, SP_NAMES) is not None
            _has_msl = _find_var(ds_model_t, MSL_NAMES) is not None
            _model_can_derive_sp = _has_sp or _has_msl
            
            # Check if model has P-E for water budget calculation
            _has_pe = _find_var(ds_model_t, ("P_minus_E_cumulative",)) is not None
            
            # If model lacks SP/MSL, we'll use Reference sp instead
            _use_ref_sp = not _model_can_derive_sp
            
            if _use_ref_sp:
                _log(f"    [{counter}] Model lacks SP/MSL — will use Reference surface pressure")
            
            if has_q and not _use_ref_sp:
                try:
                    ps_model = _get_ps(ds_model_t, ds_static_model, level_dim=model_level_dim)
                except Exception as exc:
                    _log(f"    [{counter}] ⚠ Could not derive surface pressure: {exc}")
                    _log(f"    [{counter}]   → mass/water/energy metrics will be NaN")
                    ps_model = None
            else:
                ps_model = None

            # Track metadata for output CSV
            if model_level_dim in ds_model_t.dims:
                _n_levels = ds_model_t.sizes[model_level_dim]
            _sp_method = (
                ps_model.attrs.get("derivation_method", "unknown")
                if ps_model is not None
                else "none"
            )

             # Compute area matching the model's own grid.
            # If model and Reference share the same lat grid (±1 point AND
            # matching coordinate values), reuse Reference area.  Otherwise
            # compute area from the model grid directly.
            area_model = area
            if lat_name in ds_model_t.dims:
                n_model = ds_model_t.sizes[lat_name]
                n_area = area.sizes[lat_name]
                if n_model == n_area and model_grid_matches_ref:
                    pass  # grids truly match
                elif abs(n_model - n_area) <= 1 and model_grid_matches_ref:
                    # Off by 1 (e.g. 720 vs 721) — simple trim
                    n = min(n_model, n_area)
                    area_model = area.isel({lat_name: slice(0, n)})
                    area_model = area_model.assign_coords(
                        {lat_name: ds_model_t[lat_name].values[:n]}
                    )
                else:
                    # Different grid — compute area from model grid
                    _log(f"    [{counter}] Computing area from model grid ({n_model} lats vs Reference {n_area})")
                    area_model = get_grid_cell_area(ds_model_t)

        # ---- Fetch and load Reference slice at valid time ----
        ds_ref_t = None
        ps_ref = None
        
        if mode in ("prediction", "model"):
             
             ds_ref_t = ds_ref_full.sel(time=valid_time)
             if "time" in ds_ref_t.dims:
                 ds_ref_t = ds_ref_t.isel(time=0)
             
             ds_ref_t = ds_ref_t.load()
             
             # If model lacks SP/MSL/geopotential: get Reference sp for conservation metrics
             if _use_ref_sp:
                 ref_level_dim = _detect_level_dim(ds_ref_t)
                 ps_ref = _get_ps(ds_ref_t, ds_static, level_dim=ref_level_dim)
             else:
                 # ps_ref is not needed in prediction-only mode for models with SP/MSL
                 ps_ref = None 
        
        else:
            # joint / ref mode: Load full 3D Reference for intrinsic metrics
            ds_ref_t = ds_ref_full.sel(time=valid_time)

            # Squeeze time dimension if present
            if "time" in ds_ref_t.dims:
                ds_ref_t = ds_ref_t.isel(time=0)

            ds_ref_t = ds_ref_t.load()
            
            # Surface pressure (needed for intrinsic metrics)
            ref_level_dim = _detect_level_dim(ds_ref_t)
            ps_ref = _get_ps(ds_ref_t, ds_static, level_dim=ref_level_dim)
        
        # If model lacks SP/MSL/geopotential: use Reference sp for model conservation metrics
        if _use_ref_sp and ps_ref is not None and has_q:
            # Interpolate Reference sp to model grid if needed
            if not model_grid_matches_ref and ds_model_t is not None:
                ps_model = ps_ref.interp(
                    latitude=ds_model_t.latitude,
                    longitude=ds_model_t.longitude,
                    method="linear"
                )
            else:
                ps_model = ps_ref
            ps_model.attrs["derivation_method"] = "ref_sp"
            _sp_method = "ref_sp"
            _log(f"    [{counter}] Using Reference surface pressure for conservation metrics")

    except Exception as exc:
        # Data loading failed entirely — nothing we can do
        _log(f"    [{counter}] ⚠ Data loading failed: {exc}")
        rows.append({
            "date": date_str,
            "lead_time_hours": lead_hours,
            "metric_name": "ERROR",
            "model_value": None,
            "ref_value": None,
            "n_levels": _n_levels,
            "sp_method": _sp_method,
        })
        return rows, [], [], []

    # ================================================================
    # Conservation / Stability drift metrics  +  time-series output
    # ================================================================
    # Compute the LINEAR TREND (slope per day) of mass/energy over a
    # time window [6 h … end_time].  end_time is looked up from
    # DRIFT_WINDOW_END for the current lead_td.  While iterating we
    # also collect hydrostatic_rmse and geostrophic_rmse at each step
    # for the secondary time-series CSV.
    #
    # PERFORMANCE RULES:
    #  - Evaluate every available timestep (no subsampling) for
    #    high-resolution diurnal-cycle time-series plots
    #  - Process in mini-batches of 5 to align with Zarr chunks
    #  - Never .load() the full 4-D window
    # ----------------------------------------------------------------

    ts_rows: list[dict] = []       # Will be merged into rows
    spectrum_rows: list[dict] = []  # Combined KE and Q spectra
    lr_dist_rows: list[dict] = []   # Full lapse rate distributions

    if mode in ("joint", "prediction", "model") and ds_model_full is not None and not _lead_td_mismatch:
        try:
            td_start = np.timedelta64(12, "h")  # Start at 12h to align with NeuralGCM
            td_end = DRIFT_WINDOW_END.get(lead_hours, lead_td)

            # --- Model time-series ---
            ds_pred_init = ds_model_full.sel(time=init_time)
            pred_td_dim = _detect_pred_td_dim(ds_pred_init) or "prediction_timedelta"

            # Lazy selection of timedelta window [12 h … end_time]
            ds_pred_window = ds_pred_init.sel(
                {pred_td_dim: slice(td_start, td_end)}
            )

            # Drop unneeded variables lazily (before any .load())
            _CONS_VARS = {
                "temperature", "geopotential",
                "u_component_of_wind", "v_component_of_wind",
                "2m_temperature", "t2m",
                "P_minus_E_cumulative",
            }
            if sp_ablation != "dry_hydro":
                _CONS_VARS.update({"specific_humidity", "q"})
            if sp_ablation not in ("hypsometric", "ref_sp"):
                _CONS_VARS.update({"surface_pressure", "sp"})
            if sp_ablation != "ref_sp":
                _CONS_VARS.update({"mean_sea_level_pressure", "msl"})

            drop_vars = [v for v in ds_pred_window.data_vars
                         if v.strip() not in _CONS_VARS]
            if drop_vars:
                ds_pred_window = ds_pred_window.drop_vars(drop_vars)

            # Squeeze time if still present as a dimension
            if "time" in ds_pred_window.dims:
                ds_pred_window = ds_pred_window.isel(time=0)

            # Keep every available timestep (no subsampling) so that
            # diurnal cycles are fully resolved in the time-series output.

            avail_tds = ds_pred_window[pred_td_dim].values
            if len(avail_tds) < 2:
                _log(f"    [{counter}] ⚠ Drift: <2 time steps in window — skipping")
                raise ValueError("Need ≥2 time steps for drift regression")

            hours_model = []
            dry_vals, water_vals, energy_vals = [], [], []
            pe_vals = []
            hydro_vals, geo_vals = [], []
            model_level_dim_d = _detect_level_dim(ds_pred_window)

            for td_val in avail_tds:
                snap = ds_pred_window.sel({pred_td_dim: td_val}).load()

                # Ensure (lat, lon) ordering
                if "latitude" in snap.dims and "longitude" in snap.dims:
                    sdims = list(snap.dims)
                    si_lon = sdims.index("longitude")
                    si_lat = sdims.index("latitude")
                    if si_lon < si_lat:
                        sdims[si_lon], sdims[si_lat] = sdims[si_lat], sdims[si_lon]
                        snap = snap.transpose(*sdims)

                # Compute the valid time for this timestep: init_time + td_val
                # Note: snap.time.values returns init_time, not valid_time
                snap_valid_time = init_time + td_val


                # For NeuralGCM: use Reference sp, compute both standard metrics AND P-E budget
                if _use_ref_sp:
                    # Model lacks SP/MSL/geopotential: get Reference sp for this timestep
                    try:
                        ref_snap_t = ds_ref_full.sel(time=snap_valid_time, method="nearest")
                        if "time" in ref_snap_t.dims:
                            ref_snap_t = ref_snap_t.isel(time=0)
                        ref_snap_t = ref_snap_t.load()
                        ref_ld_snap = _detect_level_dim(ref_snap_t)
                        ps_snap_ref = _get_ps(ref_snap_t, ds_static, level_dim=ref_ld_snap)
                        
                        # Interpolate to model grid if needed
                        if not model_grid_matches_ref:
                            ps_snap = ps_snap_ref.interp(
                                latitude=snap.latitude,
                                longitude=snap.longitude,
                                method="linear"
                            )
                        else:
                            ps_snap = ps_snap_ref
                        
                        # Ensure ps_snap has same spatial dimension order as snap (lat, lon)
                        if "latitude" in ps_snap.dims and "longitude" in ps_snap.dims:
                            ps_dims = list(ps_snap.dims)
                            if ps_dims.index("longitude") < ps_dims.index("latitude"):
                                ps_snap = ps_snap.transpose("latitude", "longitude")
                        
                        ps_snap.attrs["derivation_method"] = "ref_sp"
                        
                        # Compute conservation metrics using Reference sp
                        dry, water, energy = compute_conservation_scalars(
                            snap, ps_snap, area_model, z_sfc=z_sfc_model,
                            level_dim=model_level_dim_d,
                        )
                        step_sp_method = "ref_sp"
                        
                        # Also compute P-E budget if available
                        pe_var = _find_var(snap, ("P_minus_E_cumulative",))
                        if pe_var:
                            pe_step = float((area_model * snap[pe_var]).sum())
                        else:
                            pe_step = float("nan")
                            
                    except Exception as exc:
                        # Fallback if Reference sp fetch fails
                        ps_snap = None
                        dry = float("nan")
                        energy = float("nan")
                        water = float("nan")
                        pe_step = float("nan")
                        step_sp_method = "failed"
                        if verbose:
                            print(f"    [{counter}] ⚠ Reference sp fetch failed: {exc}")
                else:
                    # Standard path for models with SP/MSL/geopotential
                    try:
                        ps_snap = _get_ps(snap, ds_static_model, level_dim=model_level_dim_d)
                        dry, water, energy = compute_conservation_scalars(
                            snap, ps_snap, area_model, z_sfc=z_sfc_model,
                            level_dim=model_level_dim_d,
                        )
                        step_sp_method = ps_snap.attrs.get("derivation_method", "unknown")
                        pe_step = float("nan")  # Track A doesn't use P-E
                    except Exception:
                        # Fallback to pure pressure levels and P-E budget
                        ps_snap = None
                        dry = float("nan")    # Cannot compute dry mass without ps
                        energy = float("nan") # Cannot compute energy without ps

                        q_var = _find_var(snap, Q_NAMES)
                        pe_var = _find_var(snap, ("P_minus_E_cumulative",))

                        if q_var and pe_var:
                            pure_tcwv = compute_pure_tcwv(snap, q_name=q_var, level_dim=model_level_dim_d)
                            water = float((area_model * pure_tcwv).sum())
                            pe_step = float((area_model * snap[pe_var]).sum())
                            step_sp_method = "fixed_1000hPa_pure_levels"
                        else:
                            water = float("nan")
                            pe_step = float("nan")
                            step_sp_method = "failed"

                # Balance RMSEs (hydrostatic + geostrophic)
                try:
                    hydro = compute_hydrostatic_imbalance(
                        snap, area_model, level_dim=model_level_dim_d,
                    )
                except Exception:
                    hydro = float("nan")
                try:
                    geo = compute_geostrophic_imbalance(
                        snap, area_model, level_dim=model_level_dim_d,
                    )
                except Exception:
                    geo = float("nan")

                h = float(td_val / np.timedelta64(1, "h"))
                hours_model.append(h)
                dry_vals.append(dry)
                water_vals.append(water)
                energy_vals.append(energy)
                pe_vals.append(pe_step)
                hydro_vals.append(hydro)
                geo_vals.append(geo)

                # Append to time-series rows
                ts_rows.append({
                    "date": date_str,
                    "forecast_hour": h,
                    "dry_mass_Eg": dry,
                    "water_mass_kg": water,
                    "total_energy_J": energy,
                    "pe_cumulative_kg": pe_step,
                    "hydrostatic_rmse": hydro,
                    "geostrophic_rmse": geo,
                    "sp_method": step_sp_method,
                })

                del snap, ps_snap  # free memory immediately

            hours_model = np.array(hours_model)
            dry_vals = np.array(dry_vals)
            water_vals = np.array(water_vals)
            energy_vals = np.array(energy_vals)
            pe_vals = np.array(pe_vals)

            _log(f"    [{counter}] Drift: {len(avail_tds)} steps "
                 f"[{hours_model[0]:.0f}h–{hours_model[-1]:.0f}h]"
                 f"  sp_method={step_sp_method}")

            # Also log balance RMSEs from the target lead time snapshot
            # Include Reference RMSE at the same valid time for comparison
            ref_hydro, ref_geo = None, None
            if ds_ref_t is not None:
                ref_ld = _detect_level_dim(ds_ref_t)
                try:
                    ref_hydro = compute_hydrostatic_imbalance(
                        ds_ref_t, area, level_dim=ref_ld,
                    )
                except Exception:
                    pass
                try:
                    ref_geo = compute_geostrophic_imbalance(
                        ds_ref_t, area, level_dim=ref_ld,
                    )
                except Exception:
                    pass
            _append("hydrostatic_rmse", hydro_vals[-1], ref_hydro)
            _append("geostrophic_rmse", geo_vals[-1], ref_geo)

            # --- Reference time-series (water & energy anomalous drift) ---
            hours_ref_arr = np.array([], dtype=np.float64)
            water_vals_ref = np.array([], dtype=np.float64)
            energy_vals_ref = np.array([], dtype=np.float64)

            try:
                hours_ref = []
                water_ref_list, energy_ref_list = [], []

                for h in hours_model:
                    vt = init_time + np.timedelta64(int(h), "h")

                    ref_snap = ds_ref_full.sel(time=vt)
                    drop_e_vars = [v for v in ref_snap.data_vars if v.strip() not in _CONS_VARS]
                    if drop_e_vars:
                        ref_snap = ref_snap.drop_vars(drop_e_vars)
                    if "time" in ref_snap.dims:
                        ref_snap = ref_snap.isel(time=0)
                    ref_snap = ref_snap.load()

                    # Ensure (latitude, longitude) dim order to match area
                    if "latitude" in ref_snap.dims and "longitude" in ref_snap.dims:
                        _edims = list(ref_snap.dims)
                        _i_lon = _edims.index("longitude")
                        _i_lat = _edims.index("latitude")
                        if _i_lon < _i_lat:
                            _edims[_i_lon], _edims[_i_lat] = _edims[_i_lat], _edims[_i_lon]
                            ref_snap = ref_snap.transpose(*_edims)


                    ref_ld = _detect_level_dim(ref_snap)
                    try:
                        ps_e = _get_ps(ref_snap, ds_static, level_dim=ref_ld)
                    except Exception:
                        ps_e = None

                    if ps_e is not None:
                        _, w_e, e_e = compute_conservation_scalars(
                            ref_snap, ps_e, area, z_sfc=z_sfc,
                            level_dim=ref_ld,
                        )
                    else:
                        w_e, e_e = float("nan"), float("nan")

                    hours_ref.append(h)
                    water_ref_list.append(w_e)
                    energy_ref_list.append(e_e)
                    del ref_snap, ps_e

                hours_ref_arr = np.array(hours_ref)
                water_vals_ref = np.array(water_ref_list)
                energy_vals_ref = np.array(energy_ref_list)
            except Exception as exc:
                _log(f"    [{counter}] ⚠ Reference drift series failed: {exc}")

            # --- Compute final drift percentages via helper ---
            if len(hours_ref_arr) >= 2 and len(water_vals_ref) >= 2:
                drift = compute_drift_percentages(
                    hours_model, dry_vals, water_vals, energy_vals,
                    hours_ref_arr, water_vals_ref, energy_vals_ref,
                )
            else:
                # Reference series unavailable — compute dry-only drift
                slope_dry = compute_drift_slope(hours_model, dry_vals)
                dry_ref = float(dry_vals[0]) if len(dry_vals) > 0 else 0
                drift = {
                    "dry_mass_drift_pct_per_day": (
                        (slope_dry / dry_ref) * 100.0
                        if dry_ref != 0 and np.isfinite(slope_dry)
                        else float("nan")
                    ),
                    "water_mass_drift_pct_per_day":   float("nan"),
                    "total_energy_drift_pct_per_day": float("nan"),
                }

            for metric_name, val in drift.items():
                _append(metric_name, val)

            # --- Water budget residual drift (P-E based) ---
            # For models using Reference sp + P-E: always compute water budget drift
            # For other models: only compute when using fixed_1000hPa_pure_levels fallback
            _compute_pe_budget = (
                (_use_ref_sp and _has_pe and len(pe_vals) >= 2 and not all(np.isnan(pe_vals))) or
                (step_sp_method == "fixed_1000hPa_pure_levels" and len(pe_vals) >= 2)
            )
            if _compute_pe_budget:
                # 1. Calculate the cumulative discrepancy D(t)
                w_0 = water_vals[0]
                pe_0 = pe_vals[0]
                
                # Because the model variable is P-E, an increase means water left the atmosphere.
                # Therefore, D(t) = (W(t) - W(0)) - (-(PE(t) - PE(0)))
                # Which simplifies to: D(t) = (W(t) - W(0)) + (PE(t) - PE(0))
                discrepancy = (water_vals - w_0) + (pe_vals - pe_0)

                # 2. Calculate the linear drift slope of the discrepancy
                slope_D = compute_drift_slope(hours_model, discrepancy)
                
                # 3. Convert to %/day relative to initial column water
                if w_0 != 0 and np.isfinite(slope_D):
                    water_budget_drift_pct = (slope_D / w_0) * 100.0
                else:
                    water_budget_drift_pct = float("nan")

                _append("water_budget_drift_pct_per_day", water_budget_drift_pct)
                _log(f"    [{counter}] Water budget drift (P-E): "
                     f"{water_budget_drift_pct:.6g} %/day")

            _log(f"    [{counter}] ✓ Drift: "
                 f"dry={drift['dry_mass_drift_pct_per_day']:+.6g}%/day  "
                 f"water={drift['water_mass_drift_pct_per_day']:+.6g}%/day  "
                 f"energy={drift['total_energy_drift_pct_per_day']:+.6g}%/day")

        except Exception as exc:
            _log(f"    [{counter}] ⚠ Drift metrics failed: {exc}")

    elif mode == "ref" or mode == "reference":
        # In ref mode, we only care about intrinsic metrics computed directly on Reference
        try:
            ref_ld = _detect_level_dim(ds_ref_t)
            
            hydro = float("nan")
            try:
                hydro = compute_hydrostatic_imbalance(ds_ref_t, area, level_dim=ref_ld)
            except Exception as exc:
                _log(f"    [{counter}] ⚠ Reference hydrostatic failed: {exc}")
                
            geo = float("nan")
            try:
                geo = compute_geostrophic_imbalance(ds_ref_t, area, level_dim=ref_ld)
            except Exception as exc:
                _log(f"    [{counter}] ⚠ Reference geostrophic failed: {exc}")
            
            _append("hydrostatic_rmse", None, hydro)
            _append("geostrophic_rmse", None, geo)
            
            if ps_ref is not None:
                dry, water, energy = compute_conservation_scalars(
                    ds_ref_t, ps_ref, area, z_sfc=z_sfc, level_dim=ref_ld
                )
                _append("dry_mass_Eg", None, dry)
                _append("water_mass_kg", None, water)
                _append("total_energy_J", None, energy)
                
            _log(f"    [{counter}] ✓ Reference Intrinsic: hydro={hydro:.4g}, geo={geo:.4g}")
            
        except Exception as exc:
            _log(f"    [{counter}] ⚠ Reference intrinsic metrics failed: {exc}")


    # ---- Spectral metrics (comparative) & Lapse Rate ----
    # Only meaningful if we have Model predictions to compare against Reference
    # Skip if we had a lead-time mismatch (e.g. NeuralGCM has no 6h step)
    if mode in ("joint", "prediction", "model") and ds_model_t is not None and ds_ref_t is not None and not _lead_td_mismatch:
        try:
            # Align Reference grid to Model (720 vs 721 lat) before spectral analysis
            ds_ref_aligned = _align_ref_to_model(ds_ref_t, ds_model_t)
        except ValueError as exc:
            _log(f"    [{counter}] ⚠ Grid alignment failed (spectral/LR skipped): {exc}")
            ds_ref_aligned = None

        if ds_ref_aligned is not None:
            # --- Lapse Rate Distribution ---
            try:
                # 1. Calculate Gamma explicitly
                t_name_p = _find_var(ds_model_t, T_NAMES)
                phi_name_p = _find_var(ds_model_t, PHI_NAMES)
                t_name_r = _find_var(ds_ref_aligned, T_NAMES)
                phi_name_r = _find_var(ds_ref_aligned, PHI_NAMES)
                
                ld_p = _detect_level_dim(ds_model_t)
                ld_r = _detect_level_dim(ds_ref_aligned)
                
                def _get_gamma(ds, t_var, phi_var, ld):
                    t500 = ds[t_var].sel({ld: 500})
                    t850 = ds[t_var].sel({ld: 850})
                    z500 = ds[phi_var].sel({ld: 500})
                    z850 = ds[phi_var].sel({ld: 850})
                    return -9.80665 * (t500 - t850) / (z500 - z850) * 1000.0

                gamma_pred = _get_gamma(ds_model_t, t_name_p, phi_name_p, ld_p)
                gamma_ref = _get_gamma(ds_ref_aligned, t_name_r, phi_name_r, ld_r)

                # 2. Define region masks
                lat_p = ds_model_t.latitude
                regions = {
                    "tropics": (lat_p >= -30) & (lat_p <= 30),
                    "nh_mid": (lat_p > 30) & (lat_p <= 60),
                    "sh_mid": (lat_p >= -60) & (lat_p < -30)
                }

                bins = np.linspace(-15, 15, 61)

                # 3. Extract and histogram
                for band_key, mask in regions.items():
                    # Mask, drop NaNs, and flatten
                    g_pred_vals = gamma_pred.where(mask, drop=True).values.ravel()
                    g_ref_vals = gamma_ref.where(mask, drop=True).values.ravel()
                    g_pred_vals = g_pred_vals[~np.isnan(g_pred_vals)]
                    g_ref_vals = g_ref_vals[~np.isnan(g_ref_vals)]

                    if len(g_pred_vals) > 0 and len(g_ref_vals) > 0:
                        hist_pred, _ = np.histogram(g_pred_vals, bins=bins, density=True)
                        hist_ref, _ = np.histogram(g_ref_vals, bins=bins, density=True)
                        
                        for bi, b_val in enumerate(bins[:-1]):
                            lr_dist_rows.append({
                                "date": date_str,
                                "lead_hours": lead_hours,
                                "region": band_key,
                                "bin_edge_lower": float(b_val),
                                "freq_pred": float(hist_pred[bi]),
                                "freq_ref": float(hist_ref[bi]),
                            })
            except Exception as exc:
                _log(f"    [{counter}] ⚠ Lapse rate distribution error: {exc}")

            # --- Spectra (KE & Q) ---
            try:
                k_pred, e_pred = compute_ke_spectrum(ds_model_t, level=500.0)
                k_ref, e_ref = compute_ke_spectrum(ds_ref_aligned, level=500.0)
                
                n_min = min(len(e_pred), len(e_ref))
                k_common = k_pred[:n_min]
                e_pred_c = e_pred[:n_min]
                e_ref_c = e_ref[:n_min]

                # Effective resolution & small-scale ratio
                try:
                    eff_res_out = _find_effective_resolution(k_common, e_pred_c, e_ref_c)
                    if isinstance(eff_res_out, tuple):
                        L_eff, ratio = eff_res_out
                    else:
                        L_eff = eff_res_out
                        ratio = float("nan")
                    _append("effective_resolution_km", L_eff, None)
                    _append("small_scale_ratio",       ratio, None)
                except Exception as exc:
                    _log(f"    [{counter}] ⚠ Eff.Res error: {exc}")

                # Spectral divergence & residual
                try:
                    s_div, s_res = compute_spectral_scores(e_pred_c, e_ref_c)
                    _append("spectral_divergence", s_div, None)
                    _append("spectral_residual",   s_res, None)
                except Exception as exc:
                    _log(f"    [{counter}] ⚠ Spectral error: {exc}")

                for wi in range(n_min):
                    spectrum_rows.append({
                        "date": date_str,
                        "lead_hours": lead_hours,
                        "variable": "KE",
                        "wavenumber": int(k_pred[wi]),
                        "power_pred": float(e_pred[wi]),
                        "power_ref": float(e_ref[wi]),
                    })
                
                if extended_spectra:
                    # KE 850 hPa
                    try:
                        k_p850, e_p850 = compute_ke_spectrum(ds_model_t, level=850.0)
                        k_r850, e_r850 = compute_ke_spectrum(ds_ref_aligned, level=850.0)
                        n_min850 = min(len(e_p850), len(e_r850))
                        for wi in range(n_min850):
                            spectrum_rows.append({
                                "date": date_str,
                                "lead_hours": lead_hours,
                                "variable": "KE_850",
                                "wavenumber": int(k_p850[wi]),
                                "power_pred": float(e_p850[wi]),
                                "power_ref": float(e_r850[wi]),
                            })
                    except Exception as exc:
                        _log(f"    [{counter}] ⚠ KE 850 spectrum error: {exc}")

                    # Q spectrum
                    has_q = _find_var(ds_model_t, Q_NAMES) is not None
                    if has_q:
                        kq_pred, eq_pred = compute_q_spectrum(ds_model_t)
                        kq_ref, eq_ref = compute_q_spectrum(ds_ref_aligned)
                        nq_min = min(len(eq_pred), len(eq_ref))
                        for wi in range(nq_min):
                            spectrum_rows.append({
                                "date": date_str,
                                "lead_hours": lead_hours,
                                "variable": "Q",
                                "wavenumber": int(kq_pred[wi]),
                                "power_pred": float(eq_pred[wi]),
                                "power_ref": float(eq_ref[wi]),
                            })
                _log(f"    [{counter}] ✓ Spectra saved")

            except ImportError:
                pass
            except Exception as exc:
                _log(f"    [{counter}] ⚠ KE spectrum error: {exc}")

    # If no rows were produced at all (all metrics failed), write a
    # placeholder so the CSV retains the (date, lead_time) structure.
    if not rows:
        _log(f"    [{counter}] ⚠ All metrics failed — writing NaN placeholder row")
        _append("ALL_METRICS_FAILED", None, None)

    return rows, ts_rows, spectrum_rows, lr_dist_rows


# ============================================================================
# Sanity Check
# ============================================================================

def _sanity_check_ref(
    ds_ref_full: xr.Dataset,
    verbose: bool = True,
) -> None:
    """
    Verify that Reference .sel(time=…) returns *different* data for different
    valid times.  If all three sample slices are identical, the datetime
    resolution is wrong and results will be meaningless.
    """
    # Pick 3 sample times from the Reference dataset's own time range
    # (at ~25%, 50%, 75%) to avoid assuming a particular year range.
    times = ds_ref_full.time.values
    n = len(times)
    sample_times = [
        np.datetime64(times[n // 4], "ns"),
        np.datetime64(times[n // 2], "ns"),
        np.datetime64(times[3 * n // 4], "ns"),
    ]

    # Pick a variable that should change over time
    test_var = _find_var(ds_ref_full, T_NAMES) or _find_var(ds_ref_full, MSL_NAMES)
    if test_var is None:
        if verbose:
            print("  ⚠ Sanity check skipped: no temperature or MSL variable found.")
        return

    if verbose:
        print(f"\n  Sanity check — verifying Reference time selection on '{test_var}':")

    means = []
    for t in sample_times:
        slc = ds_ref_full[test_var].sel(time=t, method="nearest")
        actual_time = slc.time.values
        m = float(slc.mean())
        means.append(m)
        if verbose:
            print(f"    requested={t}  actual={actual_time}  mean={m:.6g}")

    if means[0] == means[1] == means[2]:
        raise RuntimeError(
            "Reference sanity check FAILED: all three sample times returned "
            "identical data.  This indicates a datetime resolution bug.  "
            f"Means: {means}"
        )

    if verbose:
        print("  ✓ Reference time selection looks correct (values differ).\n")


# ============================================================================
# Batch Analysis
# ============================================================================

def _parse_lead_times(spec: str) -> list[tuple[str, np.timedelta64]]:
    """Parse a comma-separated lead-time specification.

    Accepted formats per item: ``6h``, ``12h``, ``5d``, ``120h``, ``10d``, ``240h``.
    Returns a list of ``(label, timedelta)`` tuples.
    """
    result = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token.endswith("d"):
            days = int(token[:-1])
            td = np.timedelta64(days * 24, "h")
            result.append((token, td))
        elif token.endswith("h"):
            hours = int(token[:-1])
            td = np.timedelta64(hours, "h")
            # Use a friendly label for common values
            if hours % 24 == 0 and hours >= 48:
                label = f"{hours // 24}d"
            else:
                label = f"{hours}h"
            result.append((label, td))
        else:
            raise ValueError(f"Cannot parse lead-time token: {token!r}")
    return result


def run_evaluation(
    dates: list[str],
    output_csv: Path,
    mode: str = "joint",
    workers: int = DEFAULT_WORKERS,
    verbose: bool = True,
    prediction_zarr: str = DEFAULT_MODEL_ZARR,
    ref_zarr: str = REF_ZARR,
    model_name: str = "model",
    lead_times: list[tuple[str, np.timedelta64]] | None = None,
    static_zarr: str | None = None,
    extended_spectra: bool = False,
    sp_ablation: str = "default",
) -> pd.DataFrame:
    """
    Compute all physics metrics for each (date × lead time), parallelised.

    Parameters
    ----------
    dates : list[str]
        ISO date strings, e.g. ["2022-01-01", "2022-01-15"].
    output_csv : Path
        Destination CSV file.
    mode : str
        'joint', 'ref', or 'model'.
    workers : int
        Number of parallel threads.
    verbose : bool
        Print progress.
    model_name : str
        Model name for secondary CSV filename.
    lead_times : list of (label, timedelta), optional
        Override the default LEAD_TIMES (6h, 5d, 10d).

    Returns
    -------
    pd.DataFrame
        Long-format with columns:
        date, lead_time_hours, metric_name, model_value, ref_value,
        n_levels, sp_method.
    """
    # Build work items: list of (date, lead_label, lead_td)
    _lead_times = lead_times if lead_times is not None else LEAD_TIMES
    work_items = [
        (date_str, lead_label, lead_td)
        for date_str in dates
        for lead_label, lead_td in _lead_times
    ]
    n_combos = len(work_items)

    if verbose:
        print("\n" + "=" * 70)
        print("  PHYSICS EVALUATION — WeatherBench 2 Zarr Streaming")
        print("=" * 70)
        print(f"  Prediction : {prediction_zarr}")
        print(f"  Reference  : {ref_zarr}")
        print(f"  Dates  : {len(dates)}")
        print(f"  Lead times : {[label for label, _ in _lead_times]}")
        print(f"  Total evals: {n_combos}")
        print(f"  Workers    : {workers}")
        print(f"  Mode       : {mode}")
        print(f"  Output : {output_csv}")
        print("=" * 70)

    # ---- Open Zarr lazily (shared across threads) ----
    # For ProcessPoolExecutor, we pass paths to workers and they open datasets themselves.
    # But checking if we can open them first is good practice.
    if verbose:
        print("\n  Checking Zarr stores (anonymous) …")
    
    # Just open to check presence and print metadata
    ds_model_check = None
    if mode in ("joint", "prediction", "model"):
        ds_model_check = open_zarr_anonymous(prediction_zarr)
    
    ds_ref_check = open_zarr_anonymous(ref_zarr)

    if verbose:
        if ds_model_check is not None:
            print(f"  Model vars: {list(ds_model_check.data_vars)[:12]}")
        print(f"  Reference vars  : {list(ds_ref_check.data_vars)[:12]}")
        print(f"  Reference time dtype: {ds_ref_check.time.dtype}")
        print(f"  Reference time range: {ds_ref_check.time.values[0]} → "
              f"{ds_ref_check.time.values[-1]}")

    # ---- Sanity check: verify Reference .sel(time=…) returns different data ----
    _sanity_check_ref(ds_ref_check, verbose=verbose)

    # Sanity check: Prediction dataset must actually contain forecast lead times
    if mode in ("joint", "prediction", "model") and ds_model_check is not None:
        if _detect_pred_td_dim(ds_model_check) is None:
            raise ValueError(
                f"Prediction dataset ({prediction_zarr}) lacks a lead-time dimension "
                f"(e.g., 'prediction_timedelta' or 'step'). You might be pointing to "
                f"an analysis (t=0) dataset instead of a forecast dataset."
            )

    # ---- Parallel evaluation ----
    if verbose:
        print(f"\n  Launching {workers} processes …\n")

    all_rows: list[dict] = []
    all_ts_rows: list[dict] = []
    all_spectrum_rows: list[dict] = []
    all_lr_dist_rows: list[dict] = []
    
    # Per-task timeout (seconds).  Prevents a single stale GCS connection
    # from blocking the entire run indefinitely.
    TASK_TIMEOUT = 600  # 10 minutes

    # Use ProcessPoolExecutor for true parallelism (avoids GIL for spectral)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for idx, (date_str, lead_label, lead_td) in enumerate(work_items, 1):
            fut = pool.submit(
                _evaluate_one,
                prediction_zarr, ref_zarr, 
                date_str, lead_label, lead_td,
                idx, n_combos, mode, verbose,
                static_zarr,
                model_name,
                extended_spectra,
                sp_ablation,
            )
            futures[fut] = (idx, date_str, lead_label)

        for fut in as_completed(futures):
            idx, date_str, lead_label = futures[fut]
            try:
                summary_rows, ts_rows, spectrum_rows, lr_dist_rows = fut.result(timeout=TASK_TIMEOUT)
                all_rows.extend(summary_rows)
                all_ts_rows.extend(ts_rows)
                all_spectrum_rows.extend(spectrum_rows)
                all_lr_dist_rows.extend(lr_dist_rows)
            except TimeoutError:
                if verbose:
                    print(f"  ⚠ Task {idx} ({date_str} {lead_label}) timed out "
                          f"after {TASK_TIMEOUT}s — skipping", flush=True)
            except Exception as exc:
                if verbose:
                    print(f"  ⚠ Worker exception (task {idx}): {exc}", flush=True)

    # ---- Sort by date + lead time for clean output ----
    all_rows.sort(key=lambda r: (r["date"], r["lead_time_hours"], r["metric_name"]))

    # ---- Save summary CSV ----
    if not all_rows:
        if verbose:
            print("\n  ⚠ No successful results obtained (all workers failed or process array empty).")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if verbose:
        _print_summary(df)
        print(f"\n  ✓ Results saved → {output_csv}")

    # ---- Save Time Series CSV ----
    if all_ts_rows:
        year_str = dates[0][:4] if dates else "unknown"
        main_stem = output_csv.stem
        ref_suffix = "_ifs" if main_stem.endswith("_ifs") else ""
        ts_csv = output_csv.parent / f"time_series_{model_name}_{year_str}{ref_suffix}.csv"
        df_ts = pd.DataFrame(all_ts_rows)
        # Handle dupes due to overlaps on target intervals across runs
        df_ts.drop_duplicates(subset=["date", "forecast_hour"], inplace=True)
        df_ts.sort_values(["date", "forecast_hour"], inplace=True)
        df_ts.to_csv(ts_csv, index=False)
        if verbose:
            print(f"  ✓ Time series saved → {ts_csv}  ({len(df_ts)} rows)")

    # ---- Save Spectra CSV ----
    if all_spectrum_rows:
        year_str = dates[0][:4] if dates else "unknown"
        main_stem = output_csv.stem
        ref_suffix = "_ifs" if main_stem.endswith("_ifs") else ""
        spec_csv = output_csv.parent / f"spectra_{model_name}_{year_str}{ref_suffix}.csv"
        df_spec = pd.DataFrame(all_spectrum_rows)
        df_spec.sort_values(["variable", "date", "lead_hours", "wavenumber"], inplace=True)
        df_spec.to_csv(spec_csv, index=False)
        if verbose:
            print(f"  ✓ Combined spectra saved → {spec_csv}  ({len(df_spec)} rows)")

    # ---- Save Lapse Rate Distribution CSV ----
    if all_lr_dist_rows:
        lr_dist_csv = output_csv.parent / f"lapse_rate_dist_{model_name}_{year_str}{ref_suffix}.csv"
        df_lrd = pd.DataFrame(all_lr_dist_rows)
        df_lrd.sort_values(["date", "lead_hours", "region", "bin_edge_lower"], inplace=True)
        df_lrd.to_csv(lr_dist_csv, index=False)
        if verbose:
            print(f"  ✓ Lapse rate distributions saved → {lr_dist_csv}  ({len(df_lrd)} rows)")

    return df


# ============================================================================
# Summary
# ============================================================================

def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    ok = df[df["metric_name"] != "ERROR"]
    n_dates = ok["date"].nunique()
    n_errors = len(df[df["metric_name"] == "ERROR"])
    print(f"  Total metric rows : {len(ok)}")
    print(f"  Dates evaluated   : {n_dates}")
    print(f"  Error entries     : {n_errors}")

    if len(ok) == 0:
        print("  ⚠  No successful calculations.")
        return

    for lead_hours, lead_grp in ok.groupby("lead_time_hours"):
        print(f"\n  ── Lead time = {lead_hours} h ──")

        for metric, mgrp in lead_grp.groupby("metric_name"):
            model = mgrp["model_value"].dropna()
            ref_vals = mgrp["ref_value"].dropna()

            if len(ref_vals) > 0 and len(model) > 0:
                print(
                    f"    {metric:30s}  "
                    f"Model={model.mean():.6g}  "
                    f"Ref={ref_vals.mean():.6g}  "
                    f"Δ={model.mean() - ref_vals.mean():+.4g}"
                )
            elif len(model) > 0:
                print(
                    f"    {metric:30s}  "
                    f"Model={model.mean():.6g}"
                )
            elif len(ref_vals) > 0:
                print(
                    f"    {metric:30s}  "
                    f"Ref={ref_vals.mean():.6g}"
                )

    print("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Physics evaluation for Model vs Reference (WB2 Zarr streaming)"
    )
    parser.add_argument(
        "--year", type=int, default=2022,
        help="Year to evaluate (default: 2022). "
             "Ignored if --dates or --month is provided.",
    )
    parser.add_argument(
        "--dates", nargs="+", default=None,
        help="Dates to evaluate, e.g. 2022-01-01 2022-01-15",
    )
    parser.add_argument(
        "--month", type=str, default=None,
        help="Evaluate all days of a month, e.g. 2022-01",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--mode", type=str, choices=["joint", "ref", "reference", "prediction", "model"], default="joint",
        help="Evaluation mode (default: joint). 'prediction' mode is optimized for model evaluation.",
    )
    parser.add_argument(
        "--model", type=str, default="model",
        help="Name of the model (default: model). Used for output filename.",
    )
    parser.add_argument(
        "--prediction-zarr", type=str, default=DEFAULT_MODEL_ZARR,
        help=f"Path to prediction Zarr (default: {DEFAULT_MODEL_ZARR})",
    )
    parser.add_argument(
        "--ref-zarr", type=str, default=REF_ZARR,
        help=f"Path to Reference Zarr (default: {REF_ZARR})",
    )
    parser.add_argument(
        "--reference", type=str, choices=["era5", "ifs"], default="era5",
        help="Reference dataset: 'era5' (default) or 'ifs' (IFS HRES t=0 analysis). "
             "When 'ifs' is selected, output files get '_ifs' suffix.",
    )
    parser.add_argument(
        "--lead-times", type=str, default=None,
        help="Comma-separated lead times, e.g. '12h,5d,10d'. "
             "Default: 6h,5d,10d.",
    )
    parser.add_argument(
        "--static-zarr", type=str, default=None,
        help="Path to Zarr with static fields (geopotential_at_surface, land_sea_mask). "
             "Defaults to --ref-zarr.  Set this when --ref-zarr lacks static fields "
             "(e.g. HRES-T0).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--extended-spectra", action="store_true",
        help="Compute Q spectrum and 850hPa KE spectrum in addition to 500hPa KE spectrum.",
    )
    parser.add_argument(
        "--sp-ablation", type=str, choices=["default", "hypsometric", "ref_sp", "dry_hydro"], default="default",
        help="Ablation study for SP method by dropping specific variables.",
    )
    args = parser.parse_args()

    dates = _resolve_dates(args)

    # Handle --reference option
    ref_zarr_path = args.ref_zarr
    static_zarr = args.static_zarr
    ref_suffix = ""
    
    if args.reference == "ifs":
        # Use IFS HRES t=0 as reference instead of ERA5
        # Check if model uses low-res grid (NeuralGCM)
        if "512x256" in args.prediction_zarr or "neuralgcm" in args.model.lower() or "512x256" in args.model:
            ref_zarr_path = IFS_T0_LOWRES_ZARR
        else:
            ref_zarr_path = IFS_T0_ZARR
        # IFS HRES t=0 lacks static fields, so we need ERA5 for those
        if static_zarr is None:
            static_zarr = REF_ZARR
        ref_suffix = "_ifs"

    if args.output:
        output = Path(args.output)
    else:
        output = OUTPUT_DIR / f"physics_evaluation_{args.model}_{args.year}{ref_suffix}.csv"

    lt = _parse_lead_times(args.lead_times) if args.lead_times else None

    run_evaluation(
        dates=dates,
        output_csv=output,
        mode=args.mode,
        workers=args.workers,
        verbose=not args.quiet,
        prediction_zarr=args.prediction_zarr,
        ref_zarr=ref_zarr_path,
        model_name=args.model,
        lead_times=lt,
        static_zarr=static_zarr,
        extended_spectra=args.extended_spectra,
        sp_ablation=args.sp_ablation,
    )


if __name__ == "__main__":
    main()