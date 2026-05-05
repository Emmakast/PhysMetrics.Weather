"""
Physics Metrics Library — Verification checks for weather models

Contains 8 diagnostic metrics for evaluating an AI weather model:

  1. Global Dry Air Mass  (conservation)
  2. Global Water Mass    (stability)
  3. Global Total Energy  (stability)
  4. Effective Resolution (spectral)
  5. Spectral divergence  (spectral)
  6. Spectral residual    (spectral)
  7. Hydrostatic Balance  (balance)
  8. Geostrophic Balance  (balance)

Compatible with WeatherBench 2 variable naming.

Dependencies: numpy, xarray, pyshtools.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import xarray as xr
from scipy.stats import linregress, wasserstein_distance


# ============================================================================
# Physical Constants
# ============================================================================

GRAVITY = 9.80665           # m/s²  – standard gravity
EARTH_RADIUS = 6.371e6      # m     – mean Earth radius
C_PD = 1004.64              # J/(kg·K) – dry air specific heat (const. pressure)
C_PV = 1810.0               # J/(kg·K) – water vapour specific heat (const. pressure)
L_V = 2.501e6               # J/kg  – latent heat of vaporisation (at 0 °C)
R_DRY = 287.05              # J/(kg·K) – specific gas constant, dry air
LAPSE_RATE = 0.0065         # K/m   – standard tropospheric lapse rate
OMEGA = 7.2921e-5           # rad/s – Earth angular velocity
EXAGRAM = 1e18              # kg    – conversion factor to Exagrams
R_V = 461.5                 # J/(kg·K) – specific gas constant, water vapor


# ============================================================================
# Variable Names
# ============================================================================

def _find_var(
    ds: xr.Dataset,
    candidates: tuple[str, ...],
) -> Optional[str]:
    """Return the first variable name found in *ds*, or ``None``."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


SP_NAMES = ("surface_pressure", "sp", "ps")
MSL_NAMES = ("mean_sea_level_pressure", "msl")
Q_NAMES = ("specific_humidity", "q")
T_NAMES = ("temperature", "t")
U_NAMES = ("u_component_of_wind", "u")
V_NAMES = ("v_component_of_wind", "v")
PHI_NAMES = ("geopotential", "z")
T2M_NAMES = ("2m_temperature", "t2m")
ZSFC_NAMES = ("geopotential_at_surface", "z_sfc", "orography")

LEVEL_DIM_NAMES = ("level", "pressure_level", "plev", "isobaricInhPa")
PRED_TD_NAMES = ("prediction_timedelta", "lead_time", "step", "timedelta")


def _detect_level_dim(ds: xr.Dataset) -> str:
    """Auto-detect the name of the pressure-level dimension."""
    for name in LEVEL_DIM_NAMES:
        if name in ds.dims:
            return name
    raise ValueError(
        f"Could not automatically detect the pressure level dimension. "
        f"Looked for: {LEVEL_DIM_NAMES}. Available dims: {list(ds.dims)}"
    )


def _detect_pred_td_dim(ds: xr.Dataset) -> Optional[str]:
    """Auto-detect the name of the prediction_timedelta dimension."""
    for name in PRED_TD_NAMES:
        if name in ds.dims:
            return name
    return None


# ============================================================================
# Grids
# ============================================================================

def get_grid_cell_area(
    ds: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    earth_radius: float = EARTH_RADIUS,
) -> xr.DataArray:
    """
    Area of each grid cell on a regular lat/lon grid (m²).

    A_i = R² × Δλ × |sin(φ_north) − sin(φ_south)|
    """
    lat = ds[lat_name].values
    lon = ds[lon_name].values

    dlon = np.abs(np.diff(lon).mean())
    dlon_rad = np.deg2rad(dlon)

    # Use exact local midpoints between adjacent latitudes as cell boundaries
    lat_rad = np.deg2rad(lat)
    midpoints = (lat_rad[:-1] + lat_rad[1:]) / 2.0
    lat_s = np.empty_like(lat_rad)
    lat_n = np.empty_like(lat_rad)
    lat_s[0] = np.clip(2 * lat_rad[0] - midpoints[0], -np.pi / 2, np.pi / 2)
    lat_s[1:] = midpoints
    lat_n[:-1] = midpoints
    lat_n[-1] = np.clip(2 * lat_rad[-1] - midpoints[-1], -np.pi / 2, np.pi / 2)
    lat_s = np.clip(lat_s, -np.pi / 2, np.pi / 2)
    lat_n = np.clip(lat_n, -np.pi / 2, np.pi / 2)

    area_1d = earth_radius**2 * dlon_rad * np.abs(np.sin(lat_n) - np.sin(lat_s))
    area_2d = np.broadcast_to(area_1d[:, None], (len(lat), len(lon)))

    return xr.DataArray(
        area_2d,
        dims=[lat_name, lon_name],
        coords={lat_name: lat, lon_name: lon},
        name="grid_cell_area",
        attrs={"units": "m²"},
    )

def derive_surface_pressure(
    ds: xr.Dataset,
    ds_static: xr.Dataset,
    msl_names: tuple[str, ...] = MSL_NAMES,
    z_names: tuple[str, ...] = ZSFC_NAMES,
    gravity: float = GRAVITY,
    r_dry: float = R_DRY,
    lapse_rate: float = LAPSE_RATE,
    lat_name: str = "latitude",
) -> xr.DataArray:
    """
    Derive surface pressure using the U.S. Standard Atmosphere (1976) profile.
    
    This avoids biases over high-elevation terrain caused by local surface 
    temperature inversions.
    
        z_sfc = Φ_s / g
        P_s = P_MSL × (1 - (Γ × z_sfc) / T_0)^(g / (R_d × Γ))
        
    Where T_0 = 288.15 K (standard sea level temperature).
    """
    # Locate variables
    msl_name = _find_var(ds, msl_names)
    if msl_name is None:
        raise ValueError(f"No MSL variable found.  Tried {msl_names}. "
                         f"Available: {list(ds.data_vars)}")
    msl = ds[msl_name]

    z_name = _find_var(ds_static, z_names)
    if z_name is None:
        raise ValueError(f"No z_sfc variable found.  Tried {z_names}. "
                         f"Available: {list(ds_static.data_vars)}")
    z_sfc = ds_static[z_name]

    # Strip singleton time from static field
    for tdim in ("time", "valid_time"):
        if tdim in z_sfc.dims:
            z_sfc = z_sfc.isel({tdim: 0}, drop=True)

    # Align grids (721 vs 720 latitudes)
    if lat_name in z_sfc.dims and lat_name in msl.dims:
        n_static = z_sfc.sizes[lat_name]
        n_target = msl.sizes[lat_name]

        if abs(n_static - n_target) > 1:
            raise ValueError("Latitude size mismatch is greater than 1. Grids are incompatible.")
        # Safely drop the extra pole if there is exactly a 1-row difference
        if n_static == n_target + 1:
            z_sfc = z_sfc.sel({lat_name: msl[lat_name].values}, method="nearest")
        elif n_target == n_static + 1:
            msl = msl.sel({lat_name: z_sfc[lat_name].values}, method="nearest")

        # Automatically align coordinate values to avoid float precision issues
        if z_sfc.sizes[lat_name] == msl.sizes[lat_name]:
            z_sfc = z_sfc.assign_coords({lat_name: msl[lat_name]})
            
    lon_name_cands = [d for d in msl.dims if "lon" in d.lower()]
    lon_name = lon_name_cands[0] if lon_name_cands else "longitude"

    if lon_name in z_sfc.dims and lon_name in msl.dims:
        if z_sfc.sizes[lon_name] != msl.sizes[lon_name]:
            raise ValueError("Longitude size mismatch between z_sfc and model MSL.")
        z_sfc = z_sfc.assign_coords({lon_name: msl[lon_name]})

    # --- U.S. Standard Atmosphere calculation ---
    # Constants
    t_0 = 288.15 # Standard sea level temperature
    exponent = gravity / (r_dry * lapse_rate)
    
    # Geometric height
    z = z_sfc / gravity 
    
    sp = msl * np.power((1.0 - (lapse_rate * z) / t_0), exponent)
    
    sp.name = "surface_pressure"
    sp.attrs = {"units": "Pa",
                "long_name": "Surface pressure (US Standard Atmosphere derivation)"}
    return sp


# ============================================================================
# Shared Column Integration
# ============================================================================

def _integrate_column(
    field_3d: np.ndarray,
    levels_hpa: np.ndarray,
    ps_2d: np.ndarray,
    gravity: float = GRAVITY,
) -> np.ndarray:
    """
    Trapezoidal column integration with surface-pressure masking.

    Integrates ``(1/g) ∫₀^{Ps} field dp`` for each grid point.

    Parameters
    ----------
    field_3d : ndarray, shape (nlevels, nlat, nlon)
        The quantity to integrate (e.g. q for TCWV, energy density for TE).
    levels_hpa : ndarray, shape (nlevels,)
        Pressure levels in hPa.
    ps_2d : ndarray, shape (nlat, nlon)
        Surface pressure in Pa.
    gravity : float
        Gravitational acceleration.

    Returns
    -------
    ndarray, shape (nlat, nlon)
        The column integral in units of [field] × Pa / (m/s²).
    """
    levels_pa = levels_hpa.astype(np.float64) * 100.0
    sort_idx = np.argsort(levels_pa)
    levels_sorted = levels_pa[sort_idx]
    field_sorted = field_3d[sort_idx]

    n = len(levels_sorted)
    col = np.zeros_like(ps_2d)

    # 1. Top of Atmosphere to first pressure level
    # Assume the field value at the top level is constant up to 0 Pa.
    dp_top = np.minimum(ps_2d, levels_sorted[0]) - 0.0
    col += field_sorted[0] * dp_top

    # 2. Interior layers
    for k in range(n - 1):
        p_top = levels_sorted[k]
        p_bot = levels_sorted[k + 1]

        # Masking: Only integrate layers that actually exist above the surface
        eff_top = np.minimum(ps_2d, p_top)
        eff_bot = np.minimum(ps_2d, p_bot)
        
        # If the whole layer is below ground, dp becomes 0
        dp = np.maximum(0.0, eff_bot - eff_top)

        # Trapezoidal average of the field
        field_avg = 0.5 * (field_sorted[k] + field_sorted[k + 1])
        col += field_avg * dp

    # 3. Lowest pressure level to the actual surface
    # If surface pressure is higher than the lowest available pressure level,
    # assume the lowest level field value extends down to the surface.
    dp_bottom = np.maximum(0.0, ps_2d - levels_sorted[-1])
    col += field_sorted[-1] * dp_bottom

    col /= gravity
    return col


def _ensure_ps_2d(ps: xr.DataArray) -> np.ndarray:
    """Squeeze surface-pressure DataArray to a plain 2-D numpy array."""
    arr = ps.values
    # Squeeze all singleton dimensions first
    arr = arr.squeeze()
    if arr.ndim == 0:
        return np.array([[float(arr)]])
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Likely an un-selected level/ensemble dimension; take index 0
        warnings.warn(
            f"ps has 3-D shape {arr.shape} after squeeze; "
            f"taking slice [0] to reduce to 2-D."
        )
        return arr[0]
    raise ValueError(f"Cannot interpret ps with shape {arr.shape} as 2-D.")


def _compute_tcwv(
    ds: xr.Dataset,
    ps: xr.DataArray,
    q_name: str = "q",
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None,
) -> xr.DataArray:
    """Integrate specific humidity → TCWV (kg/m²)."""
    q = ds[q_name]
    if levels is None:
        levels = ds[level_dim].values
    ps_np = _ensure_ps_2d(ps)

    lat_dim = [d for d in q.dims if d != level_dim][0]
    lon_dim = [d for d in q.dims if d != level_dim][1]

    tcwv_np = _integrate_column(q.transpose(level_dim, ...).values, levels, ps_np)

    return xr.DataArray(
        tcwv_np,
        dims=[lat_dim, lon_dim],
        coords={lat_dim: q[lat_dim], lon_dim: q[lon_dim]},
        name="tcwv",
        attrs={"units": "kg/m²", "long_name": "Total Column Water Vapour"},
    )

# ============================================================================
# Metric 1 — Global Dry Air Mass
# ============================================================================

def compute_dry_air_mass(
    ds: xr.Dataset,
    ps: xr.DataArray,
    area: xr.DataArray,
    q_name: str = "q",
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None,
    tcwv: Optional[xr.DataArray] = None,
) -> float:
    """
    M_d = Σ A_i × (P_s,i / g − TCWV_i)     (returns Exagrams)
    """
    if tcwv is None:
        tcwv = _compute_tcwv(ds, ps, q_name=q_name,
                             level_dim=level_dim, levels=levels)
    col_dry = ps / GRAVITY - tcwv

    # Ensure shapes match to prevent broadcast errors
    if area.shape != col_dry.shape:
        raise ValueError(f"Area shape {area.shape} and data shape {col_dry.shape} do not match.")

    dry_mass_kg = float((area * col_dry).sum())
    return dry_mass_kg / EXAGRAM


# ============================================================================
# Metric 2 — Global Water Mass
# ============================================================================

def compute_water_mass(
    ds: xr.Dataset,
    ps: xr.DataArray,
    area: xr.DataArray,
    q_name: str = "q",
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None,
    tcwv: Optional[xr.DataArray] = None,
) -> float:
    """
    M_w = Σ A_i × TCWV_i      (returns kg)
    """
    if tcwv is None:
        tcwv = _compute_tcwv(ds, ps, q_name=q_name,
                             level_dim=level_dim, levels=levels)
    if area.shape != tcwv.shape:
        raise ValueError("Area and TCWV shapes do not match!")
    return float((area * tcwv).sum())


# ============================================================================
# Metric 3 — Global Total Energy
# ============================================================================

def compute_total_energy(
    ds: xr.Dataset,
    ps: xr.DataArray,
    area: xr.DataArray,
    z_sfc: xr.DataArray,
    t_name: str = "temperature",
    q_name: str = "q",
    u_names: tuple[str, ...] = U_NAMES,
    v_names: tuple[str, ...] = V_NAMES,
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None,
    c_pd: float = C_PD,
    c_pv: float = C_PV,
    l_v: float = L_V,
) -> float:
    """
    TE = (1/g) Σ A_i ∫ (c_p T + Φ_s + L_v q + ½(u² + v²)) dp   (returns J)

    Uses moist-air specific heat:  c_p = c_pd (1 − q) + c_pv q

    Uses static surface geopotential Φ_s (2-D) instead of the time-varying
    3-D geopotential on pressure levels, following the conservation formula:

        (1/g) ∫₀^{Ps} (c_p T + L_v q + Φ_s + k) dp

    Kinetic energy is always computed explicitly from the u and v wind
    components as 0.5 (u² + v²) to ensure consistency with momentum.
    A ``ValueError`` is raised if u or v are not found in the dataset.
    """
    if levels is None:
        levels = ds[level_dim].values

    # Ensure level-first ordering for all 3D numpy arrays
    T = ds[t_name].transpose(level_dim, ...).values
    q = ds[q_name].transpose(level_dim, ...).values

    # Moist-air specific heat: c_p = c_pd (1 - q) + c_pv q
    c_p = c_pd * (1.0 - q) + c_pv * q

    # Align z_sfc grid to dataset (handles 721↔720 latitude mismatch)
    ref_var = ds[t_name]
    lat_dim_ = [d for d in ref_var.dims if d != level_dim][0]
    lon_dim_ = [d for d in ref_var.dims if d != level_dim][1]
    # Safer implementation:
    lat_diff = abs(z_sfc.sizes[lat_dim_] - ref_var.sizes[lat_dim_])
    lon_diff = abs(z_sfc.sizes[lon_dim_] - ref_var.sizes[lon_dim_])

    if lat_diff <= 1 and lon_diff == 0:
        # Acceptable 721 vs 720 mismatch; safe to align
        z_aligned = z_sfc.reindex_like(ref_var.isel({level_dim: 0}), method="nearest")
    else:
        # Unacceptable grid mismatch!
        raise ValueError(
            f"Grid mismatch too large! z_sfc={z_sfc.shape}, model={ref_var.shape}. "
            "Benchmark requires native grid alignment."
        )

    # Following Sha et al. (2025), surface geopotential is used
    # uniformly throughout the column rather than integrating
    # the full vertical geopotential profile Φ(p).
    # Broadcast 2-D surface geopotential to 3-D (level, lat, lon)
    nlevels = T.shape[0]
    z_sfc_np = z_aligned.values
    z_sfc_3d = np.broadcast_to(z_sfc_np[None, :, :], (nlevels,) + z_sfc_np.shape)

    # Kinetic energy: always from u and v components
    u_var = _find_var(ds, u_names)
    v_var = _find_var(ds, v_names)
    if u_var is None or v_var is None:
        raise ValueError(
            f"u/v wind components required for KE calculation. "
            f"Tried {u_names}/{v_names}. Available: {list(ds.data_vars)}"
        )
    wspd_sq = ds[u_var].transpose(level_dim, ...).values ** 2 + ds[v_var].transpose(level_dim, ...).values ** 2

    energy_density = c_p * T + z_sfc_3d + l_v * q + 0.5 * wspd_sq
    ps_np = _ensure_ps_2d(ps)
    col_energy = _integrate_column(energy_density, levels, ps_np)
    col_da = xr.DataArray(
        col_energy,
        dims=[lat_dim_, lon_dim_],
        coords={lat_dim_: ds[lat_dim_], lon_dim_: ds[lon_dim_]},
    )
    if area.shape != col_da.shape:
        raise ValueError("Area and ENERGY shapes do not match!")
    return float((area * col_da).sum())


# ============================================================================
# Metric 4 — Effective Resolution
# ============================================================================

def _ke_spectrum_spharm(
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    KE spectrum E(l) via pyshtools SHExpandDH.

    Parameters
    ----------
    u, v : ndarray, shape (nlat, nlon)
        Wind components (N→S latitude order).

    Returns
    -------
    wavenumber, energy : ndarray
    """
    try:
        import pyshtools as pysh
    except ImportError:
        raise ImportError("pyshtools is required for metric 4. "
                          "Install with: pip install pyshtools")

    u = np.asarray(u, dtype=np.float64).squeeze()
    v = np.asarray(v, dtype=np.float64).squeeze()

    if u.ndim == 3:
        raise ValueError(
            "Expected 2D wind field for spectral analysis. "
            "You must explicitly select the 500 hPa level before passing data to this metric."
        )
    if u.ndim != 2:
        raise ValueError(f"Expected 2-D wind fields, got u.shape = {u.shape}")

    nlat, nlon = u.shape

    # pyshtools needs even-sized grids
    if nlat % 2 != 0:
        u, v = u[:-1, :], v[:-1, :]
        nlat -= 1
    if nlon % 2 != 0:
        u, v = u[:, :-1], v[:, :-1]
        nlon -= 1

    # SHExpandDH requires nlon == 2*nlat (sampling=2) or nlon == nlat (sampling=1).
    # For non-standard grids (e.g. NeuralGCM 256lat x 512lon already handled by
    # transpose; but catch any remaining mismatch), regrid to the largest valid
    # DH grid that doesn't exceed the input resolution.
    if nlon == 2 * nlat:
        sampling = 2
        lmax = nlat // 2 - 1
    elif nlon == nlat:
        sampling = 1
        lmax = nlat // 2 - 1
    else:
        raise ValueError(
            f"Grid ({nlat}, {nlon}) is not a valid Driscoll-Healy grid. "
            f"Spectral analysis requires nlon == nlat or nlon == 2*nlat."
        )

    u_c = pysh.expand.SHExpandDH(u, sampling=sampling, lmax_calc=lmax)
    v_c = pysh.expand.SHExpandDH(v, sampling=sampling, lmax_calc=lmax)

    wavenumber = np.arange(lmax + 1)
    energy = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        for m in range(l + 1):
            pw = u_c[0, l, m] ** 2 + v_c[0, l, m] ** 2
            if m > 0:
                pw += u_c[1, l, m] ** 2 + v_c[1, l, m] ** 2
            energy[l] += 0.5 * pw

    return wavenumber, energy


def _scalar_spectrum_spharm(
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Power spectrum S(l) of a scalar field via pyshtools SHExpandDH.

    S(l) = |c_{l,0}|^2 + sum_{m=1}^{l} (|c^cos_{l,m}|^2 + |c^sin_{l,m}|^2)

    Parameters
    ----------
    field : ndarray, shape (nlat, nlon)
        Scalar field (N→S latitude order).

    Returns
    -------
    wavenumber, power : ndarray
    """
    try:
        import pyshtools as pysh
    except ImportError:
        raise ImportError("pyshtools is required for spectral analysis. "
                          "Install with: pip install pyshtools")

    field = np.asarray(field, dtype=np.float64).squeeze()

    if field.ndim == 3:
        raise ValueError(
            "Expected 2D wind field for spectral analysis. "
            "You must explicitly select the 500 hPa level before passing data to this metric."
        )
    if field.ndim != 2:
        raise ValueError(f"Expected 2-D field, got shape = {field.shape}")

    nlat, nlon = field.shape

    if nlat % 2 != 0:
        field = field[:-1, :]
        nlat -= 1
    if nlon % 2 != 0:
        field = field[:, :-1]
        nlon -= 1

    if nlon == 2 * nlat:
        sampling = 2
        lmax = nlat // 2 - 1
    elif nlon == nlat:
        sampling = 1
        lmax = nlat // 2 - 1
    else:
        raise ValueError(
            f"Grid ({nlat}, {nlon}) is not a valid Driscoll-Healy grid. "
            f"Spectral analysis requires nlon == nlat or nlon == 2*nlat."
        )

    coeffs = pysh.expand.SHExpandDH(field, sampling=sampling, lmax_calc=lmax)

    wavenumber = np.arange(lmax + 1)
    power = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        for m in range(l + 1):
            pw = coeffs[0, l, m] ** 2
            if m > 0:
                pw += coeffs[1, l, m] ** 2
            power[l] += pw

    return wavenumber, power


def compute_q_spectrum(
    ds: xr.Dataset,
    level: float = 500.0,
    q_names: tuple[str, ...] = Q_NAMES,
    level_dim: str = "level",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the specific-humidity power spectrum S_q(l) at a single pressure level.

    S_q(k) = sum_{m=-k}^{k} |q_hat_{k,m}|^2

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing specific humidity.
    level : float
        Pressure level in hPa (default 500).
    q_names : tuple[str, ...]
        Candidate variable names for specific humidity.
    level_dim : str
        Name of the pressure-level dimension.

    Returns
    -------
    (wavenumber, power) : (ndarray, ndarray)
    """
    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)

    q_var = _find_var(ds, q_names)
    if q_var is None:
        raise ValueError(
            f"Specific humidity not found. Tried {q_names}. "
            f"Available: {list(ds.data_vars)}"
        )
    q = ds[q_var]
    if level_dim in q.dims:
        lvls = ds[level_dim].values
        idx = int(np.abs(lvls - level).argmin())
        q = q.isel({level_dim: idx})

    # Ensure (latitude, longitude) order
    lat_dims = [d for d in q.dims if "lat" in d.lower()]
    lon_dims = [d for d in q.dims if "lon" in d.lower()]
    if lat_dims and lon_dims:
        q = q.transpose(..., lat_dims[0], lon_dims[0])

    return _scalar_spectrum_spharm(q.values)


def _find_effective_resolution(
    k: np.ndarray,
    e_pred: np.ndarray,
    e_true: np.ndarray,
    threshold: float = 0.5,
    k_min: int = 10,
    n_consecutive: int = 5,
    earth_radius: float = EARTH_RADIUS,
) -> tuple[float, float]:
    """
    L_eff (km) = 2π R / k_cutoff where E_pred/E_true consistently < threshold.

    The cutoff wavenumber is the first k where the energy ratio drops below
    *threshold* and stays below for at least *n_consecutive* consecutive
    wavenumbers.  This avoids flagging isolated noisy dips as genuine
    resolution loss.

    Returns (L_eff_km, small_scale_ratio).
    """
    mask = (k >= k_min) & (e_true > 1e-12)
    k_sel = k[mask]
    ratio = e_pred[mask] / e_true[mask]

    below = ratio < threshold  # boolean array

    # Find the first index where *n_consecutive* consecutive values are below
    # threshold.  Use a rolling sum: if sum of n_consecutive bools == n, we
    # have a sustained drop.
    n = len(ratio)
    if n < n_consecutive:
        raise ValueError(
            f"Spectrum too short ({n} wavenumbers). Need at least {n_consecutive} "
            "to determine effective resolution."
        )

    # Rolling count of consecutive below-threshold values
    idx = None
    run = 0
    for i in range(n):
        if below[i]:
            run += 1
            if run >= n_consecutive:
                idx = i - n_consecutive + 1  # first index of the run
                break
        else:
            run = 0

    if idx is None:
        # Never sustained n_consecutive below threshold → fully resolved
        # down to the grid scale.  Cap at the Nyquist wavelength:
        #   λ_grid = 2πR / l_max  where l_max = max wavenumber in spectrum
        l_max = float(k_sel[-1]) if len(k_sel) > 0 else float(k[-1])
        L_grid_km = (2.0 * np.pi * earth_radius / l_max) / 1000.0
        return L_grid_km, float(np.mean(ratio))

    k_c = float(k_sel[idx])
    L_km = (2.0 * np.pi * earth_radius / k_c) / 1000.0

    return L_km


def compute_ke_spectrum(
    ds: xr.Dataset,
    level: float = 500.0,
    u_names: tuple[str, ...] = U_NAMES,
    v_names: tuple[str, ...] = V_NAMES,
    level_dim: str = "level",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the kinetic energy spectrum E(l) at a single pressure level.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing u and v wind components.
    level : float
        Pressure level in hPa (default 500).
    u_names, v_names : tuple[str, ...]
        Candidate variable names for wind components.
    level_dim : str
        Name of the pressure-level dimension.

    Returns
    -------
    (wavenumber, energy) : (ndarray, ndarray)
        Spherical-harmonic wavenumber and KE at each wavenumber.
    """
    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)

    u_var = _find_var(ds, u_names)
    v_var = _find_var(ds, v_names)
    if u_var is None or v_var is None:
        raise ValueError(
            f"u/v not found. Tried {u_names}/{v_names}. "
            f"Available: {list(ds.data_vars)}"
        )
    u = ds[u_var]
    v = ds[v_var]
    if level_dim in u.dims:
        lvls = ds[level_dim].values
        idx = int(np.abs(lvls - level).argmin())
        u = u.isel({level_dim: idx})
        v = v.isel({level_dim: idx})

    # Ensure (latitude, longitude) order — some models (e.g. NeuralGCM on WB2)
    # store dims as (longitude, latitude).
    for da in (u, v):
        lat_dims = [d for d in da.dims if "lat" in d.lower()]
        lon_dims = [d for d in da.dims if "lon" in d.lower()]
        if lat_dims and lon_dims:
            u = u.transpose(..., lat_dims[0], lon_dims[0])
            v = v.transpose(..., lat_dims[0], lon_dims[0])
            break

    return _ke_spectrum_spharm(u.values, v.values)


def compute_spectral_scores(
    e_pred: np.ndarray,
    e_true: np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """
    Spectral evaluation metrics from pre-computed KE spectra.

    Parameters
    ----------
    e_pred, e_true : ndarray
        1-D energy arrays (must have the same length).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    (spec_div, spec_res)

    spec_div : float
        Spectral Divergence — Wasserstein distance between the normalised energy
        distributions of the true and predicted spectra::

            P(k) = E_true(k) / sum(E_true)
            Q(k) = E_pred(k) / sum(E_pred)
            SpecDiv = Wasserstein distance

    spec_res : float
        Spectral Residual — RMSE of the log-energy difference::

            SpecRes = sqrt( (1/K) sum_k (log(E_pred+eps) - log(E_true+eps))^2 )
    """
    # Create the wavenumber array (0 to len-1)
    k = np.arange(len(e_pred))
    
    # ---- Spectral Divergence (1-Wasserstein Distance) ----
    # Scipy normalizes the weights internally, so passing raw energy is safe.
    spec_div = float(wasserstein_distance(
        u_values=k, 
        v_values=k, 
        u_weights=e_true, 
        v_weights=e_pred
    ))

    # ---- Spectral Residual (log-RMSE) ----
    log_diff = np.log(e_pred + eps) - np.log(e_true + eps)
    spec_res = float(np.sqrt(np.mean(log_diff ** 2)))

    return spec_div, spec_res


# ============================================================================
# Metric 5 — Hydrostatic Balance RMSE
# ============================================================================

def compute_hydrostatic_imbalance(
    ds: xr.Dataset,
    area: xr.DataArray,
    phi_name: str | None = None,
    t_name: str | None = None, 
    q_name: str = "q",
    level_dim: str = "level",
    lat_name: str = "latitude",
    p_top: float = 500.0,
    p_bot: float = 850.0,
    r_dry: float = R_DRY,
) -> float:
    """
    Hypsometric check between p_top and p_bot hPa.

    Error = |(Φ_top − Φ_bot) − R_d T̄_v ln(p_bot/p_top)|

    T̄_v is the mean virtual temperature of the layer, approximated as
    the mean of T_v at the two levels where T_v ≈ T(1 + 0.(R_V/R_DRY - 1) q).

    Returns area-weighted RMSE (m²/s²).
    """
    # Auto-detect level dimension if default is missing
    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)
    levels = ds[level_dim].values

    if phi_name is None:
        phi_name = _find_var(ds, PHI_NAMES)
        if phi_name is None:
            raise ValueError(f"No geopotential variable found. Tried {PHI_NAMES}. Available: {list(ds.data_vars)}")
    if t_name is None:
        t_name = _find_var(ds, T_NAMES)
        if t_name is None:
            raise ValueError(f"No temperature variable found. Tried {T_NAMES}. Available: {list(ds.data_vars)}")
    if q_name not in ds.data_vars:
        q_name = _find_var(ds, Q_NAMES) or q_name 

    def _sel_level(var, p):
        idx = int(np.abs(levels - p).argmin())
        return ds[var].isel({level_dim: idx})

    phi_top = _sel_level(phi_name, p_top)
    phi_bot = _sel_level(phi_name, p_bot)

    T_top = _sel_level(t_name, p_top)
    T_bot = _sel_level(t_name, p_bot)

    # Virtual temperature
    if q_name in ds.data_vars:
        q_top = _sel_level(q_name, p_top)
        q_bot = _sel_level(q_name, p_bot)
        Tv_top = T_top * (1.0 + (R_V/R_DRY - 1) * q_top)
        Tv_bot = T_bot * (1.0 + (R_V/R_DRY - 1) * q_bot)
    else:
        Tv_top = T_top
        Tv_bot = T_bot

    # Two-level approximation of ∫ T_v d ln p
    Tv_mean = 0.5 * (Tv_top + Tv_bot)

    # Hypsometric equation:  Φ_top − Φ_bot = R_d T̄_v ln(p_bot/p_top)
    lhs = phi_top - phi_bot
    rhs = r_dry * Tv_mean * np.log(p_bot / p_top)
    error = lhs - rhs

    # Ensure error has the same (lat, lon) dim order as area
    lat_dim_e = next((d for d in error.dims if "lat" in d.lower()), None)
    lon_dim_e = next((d for d in error.dims if "lon" in d.lower()), None)
    if lat_dim_e and lon_dim_e and error.dims != (lat_dim_e, lon_dim_e):
        error = error.transpose(lat_dim_e, lon_dim_e)

    # Area-weighted RMSE
    if area.shape != error.shape:
        raise ValueError(f"Area shape {area.shape} and error shape {error.shape} do not match in Hydrostatic check.")
    weights = area / float(area.sum())
    mse = float((weights.values * error.values**2).sum())
    return float(np.sqrt(mse))


# ============================================================================
# Metric 6 — Geostrophic Balance RMSE
# ============================================================================

def compute_geostrophic_imbalance(
    ds: xr.Dataset,
    area: xr.DataArray,
    phi_name: str = "geopotential",
    u_names: tuple[str, ...] = U_NAMES,
    v_names: tuple[str, ...] = V_NAMES,
    level: float = 500.0,
    level_dim: str = "level",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    lat_cutoff: float = 10.0,
    earth_radius: float = EARTH_RADIUS,
    omega: float = OMEGA,
) -> float:
    """
    RMSE of |V_actual − V_geostrophic| at *level* hPa.

    u_g = −(1 / fR)  ∂Φ/∂φ
    v_g =  (1 / fR cosφ)  ∂Φ/∂λ

    Latitudes within ±lat_cutoff° are excluded (equatorial singularity).
    Latitudes beyond ±89.9° are excluded (polar singularity).

    Returns area-weighted RMSE (m/s).
    """
    # Auto-detect level dimension if default is missing
    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)

    levels = ds[level_dim].values
    idx = int(np.abs(levels - level).argmin())

    # Get Φ at the chosen level
    phi_var = _find_var(ds, (phi_name,))
    if phi_var is None:
        phi_var = _find_var(ds, PHI_NAMES)
    if phi_var is None:
        raise ValueError(f"Geopotential not found. "
                         f"Available: {list(ds.data_vars)}")
    phi = ds[phi_var]
    if level_dim in phi.dims:
        phi = phi.isel({level_dim: idx})

    # Get actual wind
    u_var = _find_var(ds, u_names)
    v_var = _find_var(ds, v_names)
    if u_var is None or v_var is None:
        raise ValueError("u/v wind not found for geostrophic check.")
    u_actual = ds[u_var]
    v_actual = ds[v_var]
    if level_dim in u_actual.dims:
        u_actual = u_actual.isel({level_dim: idx})
        v_actual = v_actual.isel({level_dim: idx})

    # Ensure 2-D by transposing to (lat, lon) order explicitly
    if lat_name in phi.dims and lon_name in phi.dims:
        phi = phi.transpose(lat_name, lon_name)
        u_actual = u_actual.transpose(lat_name, lon_name)
        v_actual = v_actual.transpose(lat_name, lon_name)

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Coriolis parameter  f = 2Ω sin(φ)
    f_1d = 2.0 * omega * np.sin(lat_rad)
    f_2d = f_1d[:, None] * np.ones((1, len(lon)))

    cos_lat = np.cos(lat_rad)
    cos_2d = cos_lat[:, None] * np.ones((1, len(lon)))

    # Geopotential gradient (central differences on sphere)
    phi_np = phi.values  # (lat, lon)

    # Enforce monotonic latitude ordering
    assert np.all(np.diff(lat_rad) > 0) or np.all(np.diff(lat_rad) < 0), (
        "Latitude must be monotonically ordered for finite differences."
    )

    # ∂Φ/∂φ  (latitude gradient)
    dPhi_dphi = np.gradient(phi_np, lat_rad, axis=0, edge_order=2)  # m²/s² per radian

    # ∂Φ/∂λ  (longitude gradient)
    phi_padded = np.pad(phi_np, pad_width=((0, 0), (1, 1)), mode="wrap")
    dlon_rad = np.abs(lon_rad[1] - lon_rad[0])
    dPhi_dlam = np.gradient(phi_padded, dlon_rad, axis=1)[:, 1:-1]

    # Geostrophic wind
    #   u_g = −(1 / (f R)) ∂Φ/∂φ
    #   v_g =  (1 / (f R cosφ)) ∂Φ/∂λ
    with np.errstate(divide="ignore", invalid="ignore"):
        u_g = -dPhi_dphi / (f_2d * earth_radius)
        v_g = dPhi_dlam / (f_2d * earth_radius * cos_2d)

    # Vector difference
    du = u_actual.values - u_g
    dv = v_actual.values - v_g
    vec_err_sq = du**2 + dv**2

    # Mask equatorial band AND poles (singularity at cos(lat)=0)
    lat_mask = (np.abs(lat) >= lat_cutoff) & (np.abs(lat) < 89.9)
    mask_2d = lat_mask[:, None] * np.ones((1, len(lon)), dtype=bool)
    vec_err_sq = np.where(mask_2d, np.nan_to_num(vec_err_sq, nan=0.0), 0.0)

    # Area-weighted RMSE (excluding equator)
    # Align area to match data grid
    area_vals = area.values
    if area_vals.shape != vec_err_sq.shape:
        # Recompute area from the actual data grid
        area_da = get_grid_cell_area(ds)
        if lat_name in area_da.dims and lon_name in area_da.dims:
            area_da = area_da.transpose(lat_name, lon_name)
        area_vals = area_da.values
    w = area_vals.copy()
    w[~mask_2d] = 0.0
    w_sum = w.sum()
    if w_sum == 0:
        return float("nan")
    w_norm = w / w_sum
    if w_norm.shape != vec_err_sq.shape:
        raise ValueError(f"Weights shape {w_norm.shape} and error shape {vec_err_sq.shape} do not match in Geostrophic check.")

    mse = float((w_norm * vec_err_sq).sum())
    return float(np.sqrt(mse))

# ============================================================================
# Metric 7 - Lapse rate calculation
# ============================================================================

def compute_lapse_rate_wasserstein(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    area: xr.DataArray,
    t_name: str | None = None,
    phi_name: str | None = None,
    level_dim_pred: str | None = None,
    level_dim_ref: str | None = None,
) -> dict[str, float]:
    """
    Computes the 1D Wasserstein distance (W1) of the environmental lapse rate 
    (between 500 hPa and 850 hPa) for three geographical bands. The calculation 
    is area-weighted to account for poleward grid cell convergence.
    """
    t_name_p = t_name or _find_var(ds_pred, T_NAMES)
    phi_name_p = phi_name or _find_var(ds_pred, PHI_NAMES)
    t_name_r = t_name or _find_var(ds_ref, T_NAMES)
    phi_name_r = phi_name or _find_var(ds_ref, PHI_NAMES)

    ld_p = level_dim_pred or _detect_level_dim(ds_pred)
    ld_r = level_dim_ref or _detect_level_dim(ds_ref)

    def _calc_gamma(ds, t_var, phi_var, ld):
        t_500 = ds[t_var].sel({ld: 500})
        t_850 = ds[t_var].sel({ld: 850})
        phi_500 = ds[phi_var].sel({ld: 500})
        phi_850 = ds[phi_var].sel({ld: 850})
        
        # Lapse rate formulation, units converted to K/km
        return -GRAVITY * (t_500 - t_850) / (phi_500 - phi_850) * 1000.0

    gamma_p = _calc_gamma(ds_pred, t_name_p, phi_name_p, ld_p)
    gamma_r = _calc_gamma(ds_ref, t_name_r, phi_name_r, ld_r)

    lat_p = ds_pred.latitude
    lat_r = ds_ref.latitude

    bands = {
        "tropics": ((lat_p >= -30) & (lat_p <= 30), (lat_r >= -30) & (lat_r <= 30)),
        "nh_mid": ((lat_p > 30) & (lat_p <= 60), (lat_r > 30) & (lat_r <= 60)),
        "sh_mid": ((lat_p >= -60) & (lat_p < -30), (lat_r >= -60) & (lat_r < -30)),
    }

    # Re-calculate native areas to avoid cross-dataset float coordinate alignment issues
    area_p = get_grid_cell_area(ds_pred)
    area_r = get_grid_cell_area(ds_ref)

    results = {}
    for band_name, (mask_p, mask_r) in bands.items():
        # Mask arrays by band and flatten to 1D
        gp_band = gamma_p.where(mask_p, drop=True).values.flatten()
        gr_band = gamma_r.where(mask_r, drop=True).values.flatten()
        
        area_p_band = area_p.where(mask_p, drop=True).values.flatten()
        area_r_band = area_r.where(mask_r, drop=True).values.flatten()

        # Isolate valid data points (exclude NaNs introduced by the mask padding)
        valid_p = ~np.isnan(gp_band) & ~np.isnan(area_p_band)
        valid_r = ~np.isnan(gr_band) & ~np.isnan(area_r_band)

        w1 = wasserstein_distance(
            u_values=gp_band[valid_p],
            v_values=gr_band[valid_r],
            u_weights=area_p_band[valid_p],
            v_weights=area_r_band[valid_r]
        )
        results[f"lapse_rate_w1_{band_name}"] = float(w1)

    return results

# ============================================================================
# Drift (Linear Trend) Helpers
# ============================================================================

def compute_drift_slope(
    hours: np.ndarray,
    values: np.ndarray,
) -> float:
    """
    Compute the linear-regression slope of *values* vs *hours*.

    Parameters
    ----------
    hours : ndarray, shape (N,)
        Lead times in hours (e.g. [6, 12, 18, …]).
    values : ndarray, shape (N,)
        Globally integrated scalar at each lead time (e.g. dry mass in Eg).

    Returns
    -------
    float
        Slope expressed as change **per day** (i.e. the hours are converted
        to fractional days before regression).
    """
    days = np.asarray(hours, dtype=np.float64) / 24.0
    vals = np.asarray(values, dtype=np.float64)

    # Need at least 2 finite points for a slope
    mask = np.isfinite(days) & np.isfinite(vals)
    if mask.sum() < 2:
        return float("nan")

    result = linregress(days[mask], vals[mask])
    return float(result.slope)


def compute_drift_percentages(
    hours_model: np.ndarray,
    dry_model: np.ndarray,
    water_model: np.ndarray,
    energy_model: np.ndarray,
    hours_ref: np.ndarray,
    water_ref: np.ndarray,
    energy_ref: np.ndarray,
) -> dict[str, float]:
    """
    Compute percentage drift rates from pre-collected time-series.

    Parameters
    ----------
    hours_model : ndarray  – lead-times in hours for Model snapshots.
    dry_model   : ndarray  – global dry-air mass (Eg) at each Model step.
    water_model : ndarray  – global water mass (kg) at each Model step.
    energy_model: ndarray  – global total energy (J) at each Model step.
    hours_ref    : ndarray  – lead-times in hours for Reference snapshots.
    water_ref    : ndarray  – global water mass (kg) at each Reference step.
    energy_ref   : ndarray  – global total energy (J) at each Reference step.

    Returns
    -------
    dict with keys ``dry_mass_drift_pct_per_day``,
    ``water_mass_drift_pct_per_day``, ``total_energy_drift_pct_per_day``.

    Dry mass uses internal drift:
        (slope_model / model_ref) * 100   (%/day)

    Water mass and energy use anomalous drift, where each trend is
    normalised by its own starting value before taking the difference:
        (slope_model / model_ref - slope_ref / ref_0) * 100   (%/day)

    This is fairer than dividing the slope difference by a single
    reference, because each source is measured on its own scale.

    Reference values are taken from the first element (≈ 6 h).
    """
    slope_dry    = compute_drift_slope(hours_model, dry_model)
    slope_water  = compute_drift_slope(hours_model, water_model)
    slope_energy = compute_drift_slope(hours_model, energy_model)

    slope_water_ref  = compute_drift_slope(hours_ref, water_ref)
    slope_energy_ref = compute_drift_slope(hours_ref, energy_ref)

    dry_ref_0    = float(dry_model[0])
    water_model_0 = float(water_model[0])
    water_ref_0 = float(water_ref[0])
    energy_model_0 = float(energy_model[0])
    energy_ref_0 = float(energy_ref[0])

    def _safe_rel_pct(slope, ref):
        """Compute (slope / ref) * 100, returning NaN on bad inputs."""
        if ref != 0 and np.isfinite(slope) and np.isfinite(ref):
            return (slope / ref) * 100.0
        return float("nan")

    return {
        "dry_mass_drift_pct_per_day":     _safe_rel_pct(slope_dry, dry_ref_0),
        "water_mass_drift_pct_per_day":   _safe_rel_pct(slope_water, water_model_0) - _safe_rel_pct(slope_water_ref, water_ref_0),
        "total_energy_drift_pct_per_day": _safe_rel_pct(slope_energy, energy_model_0) - _safe_rel_pct(slope_energy_ref, energy_ref_0),
    }


def compute_conservation_scalars(
    ds: xr.Dataset,
    ps: xr.DataArray,
    area: xr.DataArray,
    z_sfc: xr.DataArray,
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None,
) -> tuple[float, float, float]:
    """
    Compute the three conservation scalars for a single 3-D snapshot.

    Returns
    -------
    (dry_mass_Eg, water_mass_kg, total_energy_J)
    """
    # Auto-detect level dimension if default is missing
    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)
    if levels is None and level_dim in ds.coords:
        levels = ds[level_dim].values

    q_name = _find_var(ds, Q_NAMES)
    t_name = _find_var(ds, T_NAMES) or "temperature"
    has_q = q_name is not None

    if has_q:
        tcwv = _compute_tcwv(ds, ps, q_name=q_name,
                             level_dim=level_dim, levels=levels)
        dry = compute_dry_air_mass(ds, ps, area, q_name=q_name,
                                   level_dim=level_dim, levels=levels, tcwv=tcwv)
        water = compute_water_mass(ds, ps, area, q_name=q_name,
                                   level_dim=level_dim, levels=levels, tcwv=tcwv)
        try:
            energy = compute_total_energy(
                ds, ps, area, z_sfc=z_sfc,
                t_name=t_name, q_name=q_name,
                level_dim=level_dim, levels=levels,
            )
        except Exception:
            energy = float("nan")
    else:
        dry = float("nan")
        water = float("nan")
        energy = float("nan")

    return dry, water, energy


# ============================================================================
# Pure Fixed-Level TCWV (no surface pressure)
# ============================================================================

def compute_pure_tcwv(
    ds: xr.Dataset,
    q_name: str = "q",
    level_dim: str = "level",
) -> xr.DataArray:
    """
    Integrate specific humidity purely over fixed pressure levels (no ps masking).

    Trapezoidal rule: TCWV = (1/g) Σ 0.5·(q_k + q_{k+1}) · Δp_k
    The highest pressure level (e.g. 1000 hPa) is treated as the column bottom.
    """
    q = ds[q_name]
    levels = ds[level_dim].values
    levels_pa = levels.astype(np.float64) * 100.0
    sort_idx = np.argsort(levels_pa)
    levels_sorted = levels_pa[sort_idx]
    q_sorted = q.transpose(level_dim, ...).values[sort_idx]

    # Standard trapezoidal integration W = (1/g) * sum(q * dp)
    col = np.zeros_like(q_sorted[0])
    for k in range(len(levels_sorted) - 1):
        dp = levels_sorted[k + 1] - levels_sorted[k]
        q_avg = 0.5 * (q_sorted[k] + q_sorted[k + 1])
        col += q_avg * dp

    lat_dim = [d for d in q.dims if d != level_dim][0]
    lon_dim = [d for d in q.dims if d != level_dim][1]

    return xr.DataArray(
        col / GRAVITY,
        dims=[lat_dim, lon_dim],
        coords={lat_dim: q[lat_dim], lon_dim: q[lon_dim]},
        name="tcwv_pure",
        attrs={"units": "kg/m²",
               "long_name": "TCWV (fixed pressure levels, no surface pressure)"},
    )
