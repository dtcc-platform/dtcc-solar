import math
import numpy as np
import pandas as pd
import pprint as pp
from dtcc_solar.logging import info, debug, warning, error
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.dome import Dome
from dtcc_solar.coefficients import calc_perez_coeffs
from dtcc_solar.utils import SkyResults, SunResults, SolarParameters


"""
Perez sky luminance distribution model implementation.
This module implements the Perez model for sky luminance distribution,
which is used to calculate the luminance of different patches of the sky
based on the position of the sun and sky conditions.
It includes functions to compute sky clearness, zenith luminance,
relative luminance, and absolute luminance for sky patches.
It also provides a data class to store the results of the calculations.
"""


def compute_sky_clearness_old(dni, dhi, sun_zenith_rad):
    """
    Compute Perez sky clearness ε for the sky luminance distribution model.

    Parameters:
    - dni: Direct normal irradiance (W/m²)
    - dhi: Diffuse horizontal irradiance (W/m²)
    - sun_zenith_rad: Solar zenith angle in radians

    Returns:
    - epsilon: Sky clearness (unitless)
    """
    if dhi <= 0:
        return float("inf")  # Defined as perfectly clear in the model

    epsilon = (dni / dhi + 1.041 * sun_zenith_rad**3) / (1 + 1.041 * sun_zenith_rad**3)

    # Clamp epsilon to the range [1.0, 12.0] as per Perez model
    epsilon = max(1.0, min(epsilon, 12.0))

    return epsilon


def compute_sky_clearness(dni, dhi, sun_zenith_rad):
    if dhi <= 0:
        return 12.0  # won't be used anyway when DHI<=0
    z = sun_zenith_rad
    eps = ((dni + dhi) / dhi + 1.041 * z**3) / (1 + 1.041 * z**3)
    # Radiance clamps
    return min(11.9, max(1.0, eps))


def calculate_air_mass_old(sun_zenith):
    """
    Calculates the relative air mass using Kasten, F. 1966.
    Input:
        sun_zenith: solar zenith angle in radians
    Returns:
        air mass (unitless)
    """
    if math.degrees(sun_zenith) >= 90:
        return float("inf")  # sun below horizon
    z = sun_zenith

    m = 1.0 / (math.cos(z) + 0.15 * pow((93.885 - math.degrees(z)), -1.253))

    m = min(m, 10)  # Clamp to a maximum value

    return m


def calculate_air_mass(sz):
    return 1.0 / (math.cos(sz) + 0.15 * (93.885 - math.degrees(sz)) ** -1.253)


def calc_julian_day(ts: pd.Timestamp):
    """
    Calculate the Julian day of the year from a date.

    Parameters:
    - date: datetime object

    Returns:
    - int: Julian day (1 to 365 or 366)
    """
    # Convert date to Julian day
    first_day = pd.Timestamp(ts.year, 1, 1, tz=ts.tz)
    delta = ts - first_day

    return delta.days + 1  # Julian day starts from 1, not 0


def calc_eccentricity(julian_day):
    """
    Calculate the Earth orbit eccentricity correction factor (E0)
    for a given Julian day.

    Reference: Sen, Z. (2008). Solar Energy Fundamentals and Modeling Techniques. Springer, p. 72.

    Parameters:
    - julian_day (int): Julian day of the year (1 to 365 or 366)

    Returns:
    - float: Eccentricity correction factor (unitless)
    """
    # Day angle in radians
    day_angle = (julian_day - 1) * (2.0 * math.pi / 365.0)

    # Eccentricity correction factor
    E0 = (
        1.00011
        + 0.034221 * math.cos(day_angle)
        + 0.00128 * math.sin(day_angle)
        + 0.000719 * math.cos(2.0 * day_angle)
        + 0.000077 * math.sin(2.0 * day_angle)
    )

    return E0


def compute_sky_brightness_old(dhi, m, epsilon, ts: pd.Timestamp, ext=1367):
    """
    Computes the sky brightness Δ.
    Inputs:
        dhi: Diffuse horizontal irradiance (W/m²)
        m: Relative air mass (unitless)
        ts: Timestamp of the calculation
        ext: Extraterrestrial solar irradiance (W/m²), default 1367
    Returns:
        Brightness Δ (unitless)
    """
    julian_day = calc_julian_day(ts)
    E0 = calc_eccentricity(julian_day)
    delta = (m * dhi) / (ext * E0)

    if epsilon < 1.065:
        delta = min(delta, 1.5)

    # Clamping found in radiance code
    # if epsilon > 1.065 and epsilon < 2.8:
    #    if delta < 0.2:
    #        delta = 0.2

    return delta


def compute_sky_brightness(dhi, m, epsilon, ts, ext=1367):
    E0 = calc_eccentricity(calc_julian_day(ts))
    delta = (m * dhi) / (ext * E0)
    # Radiance-style clamps
    delta = min(0.6, max(0.01, delta))
    if 1.065 < epsilon < 2.8 and delta < 0.2:
        delta = 0.2
    return delta


def perez_rel_lum(ksi, gamma, A, B, C, D, E):
    """
    Computes the Perez relative luminance distribution function F(θ, γ).

    Inputs:
        ksi: Zenith angle θ for sky patch (radians)
        gamma: Angle between sky patch and sun (radians)
        A-E: Perez coefficients
    Returns:
        Relative luminance factor F (unitless)
    """

    # Ensure cos_ksi is not too small to avoid numerical issues
    cos_ksi = max(math.cos(ksi), 1e-4)

    term1 = 1 + A * math.exp(B / cos_ksi)
    term2 = 1 + C * math.exp(D * gamma) + E * math.cos(gamma) ** 2

    f = term1 * term2

    f = max(f, 0.0)  # Ensure non-negative luminance

    return f


def calc_sky_sun_matrices(
    sunpath: Sunpath,
    skydome: Dome,
    sundome: Dome = None,
) -> list[SkyResults, SunResults]:

    if sundome is None:
        info("Calculating sun matrix from NaturalSunDome geometry...")
        sun_res = calc_sun_matrix_from_sunpath(sunpath)
    else:
        sun_res = calc_sun_matrix_from_dome_fast(sunpath, sundome)

    sky_res = calc_sky_matrix(sunpath, skydome)
    calc_tot_error(sunpath, sky_res, skydome, sun_res, sundome)
    return (sky_res, sun_res)


def calc_sky_matrix(
    sunpath: Sunpath, skydome: Dome, store_angles: bool = False
) -> SkyResults:
    dni = np.asarray(sunpath.sunc.dni, dtype=float)
    dhi = np.asarray(sunpath.sunc.dhi, dtype=float)
    sun_vecs = np.asarray(sunpath.sunc.sun_vecs, dtype=float)
    sun_zenith = np.asarray(sunpath.sunc.zeniths, dtype=float)
    sun_times = pd.DatetimeIndex(sunpath.sunc.time_stamps)

    # Sky patch constants
    ray_dirs = np.asarray(skydome.ray_dirs, dtype=float)  # (P,3)
    solid_angles = np.asarray(skydome.solid_angles, dtype=float)  # (P,)
    ksis = np.asarray(skydome.patch_zeniths, dtype=float)  # (P,)

    cos_ksi = np.cos(ksis)
    cos_ksi_safe = np.maximum(cos_ksi, 1e-4)  # for exp(B/cos)
    dome_solid = float(np.sum(solid_angles))

    P = ray_dirs.shape[0]
    T = sun_vecs.shape[0]

    rel_lum = np.zeros((P, T), dtype=np.float32)
    sky_mat = np.zeros((P, T), dtype=np.float32)

    if store_angles:
        all_ksis = np.zeros((P, T), dtype=np.float32)
        all_gammas = np.zeros((P, T), dtype=np.float32)
        all_ksis[:] = ksis[:, None]  # constant per patch
    else:
        all_ksis = None
        all_gammas = None

    zenith_limit = np.deg2rad(89.9)
    norm_limit = 0.01

    # Precompute E0 per time (avoids per-iteration Timestamp maths)
    doy = sun_times.dayofyear.to_numpy()
    day_angle = (doy - 1.0) * (2.0 * np.pi / 365.0)
    E0 = (
        1.00011
        + 0.034221 * np.cos(day_angle)
        + 0.00128 * np.sin(day_angle)
        + 0.000719 * np.cos(2.0 * day_angle)
        + 0.000077 * np.sin(2.0 * day_angle)
    )

    valid = (dhi > 0.0) & (sun_zenith < zenith_limit)
    valid_idx = np.where(valid)[0]

    small_norms = 0
    eval_count = 0
    ignored_dhi = float(np.sum(dhi[~valid]))

    for i in valid_idx:
        z = float(sun_zenith[i])
        dhi_i = float(dhi[i])
        dni_i = float(dni[i])

        # air mass (scalar)
        # (you can still clamp if you want; this matches your newer version)
        m = 1.0 / (math.cos(z) + 0.15 * (93.885 - math.degrees(z)) ** -1.253)

        # epsilon (scalar)
        eps = ((dni_i + dhi_i) / dhi_i + 1.041 * z**3) / (1.0 + 1.041 * z**3)
        eps = min(11.9, max(1.0, eps))

        # delta (scalar) with Radiance-style clamps
        delta = (m * dhi_i) / (1367.0 * float(E0[i]))
        delta = min(0.6, max(0.01, delta))
        if 1.065 < eps < 2.8 and delta < 0.2:
            delta = 0.2

        A, B, C, D, E = calc_perez_coeffs(eps, delta, z)

        # Vectorised over patches
        sv = sun_vecs[i]  # (3,)
        dots = ray_dirs @ sv  # (P,)
        dots = np.clip(dots, -1.0, 1.0)

        gamma = np.arccos(dots)
        gamma = np.clip(gamma, 1e-4, np.pi)

        # Perez relative luminance
        term1 = 1.0 + A * np.exp(B / cos_ksi_safe)
        term2 = 1.0 + C * np.exp(D * gamma) + E * (dots * dots)  # cos^2(gamma)=dot^2
        lvs = term1 * term2
        lvs = np.maximum(lvs, 0.0)

        # normalisation (Perez eq)
        norm = float(np.sum(lvs * cos_ksi * solid_angles))

        if norm <= norm_limit:
            small_norms += 1
            Rvs = np.full(P, dhi_i / np.pi, dtype=np.float64)
        else:
            Rvs = (lvs * dhi_i) / norm

        rel_lum[:, i] = lvs.astype(np.float32)
        sky_mat[:, i] = Rvs.astype(np.float32)

        if store_angles:
            all_gammas[:, i] = gamma.astype(np.float32)

        eval_count += 1

    info("-----------------------------------------------------")
    info("Sky matrix calculation summary (Perez):")
    info(f"  Evaluated {eval_count} sun positions of {T} which passed the checks.")
    info(f"  Conditions: dhi > 0 and sun zenith < {math.degrees(zenith_limit)} °")
    info(f"  For {small_norms} cases the norm factor <  {norm_limit} => uniform sky")
    info("-----------------------------------------------------")

    res = SkyResults()
    res.count = T
    res.relative_luminance = rel_lum
    res.solid_angles = solid_angles
    res.matrix = sky_mat
    res.ksis = all_ksis
    res.gammas = all_gammas
    res.ignored_dhi = ignored_dhi
    return res


def calc_tot_error(
    sunp: Sunpath,
    sky_res: SkyResults,
    skydome: Dome,
    sun_res: SunResults,
    sundome: Dome = None,
):

    sky_cos_zen = np.cos(np.array(skydome.patch_zeniths))
    sky_solid_angles = np.array(skydome.solid_angles)

    if sundome is None:
        sun_solid_angles = np.ones(len(sunp.sunc.sun_vecs), dtype=np.float32)
    else:
        sun_solid_angles = np.array(sundome.solid_angles)

    sun_dni = np.sum(np.sum(sun_res.matrix, axis=1) * sun_solid_angles)
    sky_dhi = np.sum(np.sum(sky_res.matrix, axis=1) * sky_cos_zen * sky_solid_angles)

    epw_dni = np.sum(sunp.sunc.dni)
    epw_dhi = np.sum(sunp.sunc.dhi)

    if epw_dni > 0:
        error_dni = np.abs(sun_dni - epw_dni) / epw_dni

    if epw_dhi > 0:
        error_dhi = np.abs(sky_dhi - epw_dhi) / epw_dhi
        error_ign = sky_res.ignored_dhi / epw_dhi

    info("-----------------------------------------------------")
    info("Comparing irradiance from weather data with sky and sun matrices:")
    info(f"  Total error in DNI: {100 * error_dni:.3f} %")
    info(f"  Total error in DHI: {100 * error_dhi:.3f} %")
    info(f"  Total ignored DHI from suns that were removed: {100 * error_ign:.3f} %")
    info("-----------------------------------------------------")


def calc_sun_matrix_from_dome_fast(sunpath: Sunpath, sundome: Dome) -> SunResults:
    sun_vecs = np.asarray(sunpath.sunc.sun_vecs, dtype=float)  # (T,3)
    dni = np.asarray(sunpath.sunc.dni, dtype=float)  # (T,)
    ray_dirs = np.asarray(sundome.ray_dirs, dtype=float)  # (P,3)
    solid = np.asarray(sundome.solid_angles, dtype=float)  # (P,)

    # Normalise (safe)
    sun_vecs /= np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True)

    # Closest patch per timestep (geometry)
    dots = ray_dirs @ sun_vecs.T  # (P,T)
    idx = np.argmax(dots, axis=0).astype(np.int32)  # (T,)

    P, T = ray_dirs.shape[0], sun_vecs.shape[0]
    sun_matrix = np.zeros((P, T), dtype=np.float32)

    # ----------------------------
    # ACTIVE SUN INDICES (geometry-only, PER TIMESTEP)
    # ----------------------------
    zen = np.asarray(sunpath.sunc.zeniths, dtype=float)  # (T,)
    sun_up = zen < np.deg2rad(90.0)

    # This is what C++ needs: length T, one patch index per timestep, -1 if sun below horizon
    active_sun_indices = np.full(T, -1, dtype=np.int32)
    active_sun_indices[sun_up] = idx[sun_up]

    # Optional debug: how many unique patches did we hit?
    active_unique = np.unique(active_sun_indices[active_sun_indices >= 0])
    info(f"Sun-up timesteps: {int(sun_up.sum())} / {T}")
    info(f"Unique active sun patches (geometry): {active_unique.size}")

    # ----------------------------
    # SUN MATRIX (DNI-based, but only when DNI > 0)
    # ----------------------------
    valid_dni = dni > 0.0
    t_idx = np.where(valid_dni)[0]
    p_idx = idx[valid_dni]  # OK even if sun below horizon; dni should be 0 there anyway
    sun_matrix[p_idx, t_idx] = (dni[valid_dni] / solid[p_idx]).astype(np.float32)

    res = SunResults(matrix=sun_matrix)
    res.active_idx = active_sun_indices  # IMPORTANT: store per-timestep indices (T,)
    return res


def calc_sun_matrix_from_dome(sunpath: Sunpath, sundome: Dome) -> SunResults:

    sun_vecs = sunpath.sunc.sun_vecs
    sun_matrix = np.zeros([len(sundome.ray_dirs), len(sun_vecs)])
    ray_dirs = np.array(sundome.ray_dirs)

    for i in range(len(sun_vecs)):
        patch_index = find_closest_patch(sun_vecs[i], ray_dirs)
        patch_solid_angle = sundome.solid_angles[patch_index]
        sun_matrix[patch_index, i] = sunpath.sunc.dni[i] / patch_solid_angle  # W/m²/sr

    sun_results = SunResults()
    sun_results.matrix = sun_matrix

    return sun_results


def calc_sun_matrix_rad(
    sunpath: Sunpath, skydome: Dome, n_targets: int = 4
) -> SunResults:
    """
    Radiance-consistent sun discretization:
      - Find the num closest patches to the sun direction
      - Weight by 1/(1.002 - dot), normalize by sum of weights
      - Convert DNI [W/m²] to radiance [W/m²/sr] by dividing with Ω_patch
    """
    ray_dirs = np.asarray(skydome.ray_dirs, dtype=float)
    solid_angles = np.asarray(skydome.solid_angles, dtype=float)
    sun_vecs = np.asarray(sunpath.sunc.sun_vecs, dtype=float)
    dni = np.asarray(sunpath.sunc.dni, dtype=float)

    ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    sun_vecs /= np.linalg.norm(sun_vecs, axis=1, keepdims=True)

    patch_count = ray_dirs.shape[0]
    n_times = sun_vecs.shape[0]
    n_patches = max(1, min(n_targets, patch_count))

    sun_matrix = np.zeros((patch_count, n_times), dtype=float)

    tot_irr = 0.0

    for t in range(n_times):
        if dni[t] <= 0.0:
            continue

        dot = np.dot(ray_dirs, sun_vecs[t])
        dot = np.clip(dot, -1.0, 1.0)

        idx = np.argpartition(-dot, n_patches)[:n_patches]

        w = 1.0 / (1.002 - dot[idx])
        w_sum = np.sum(w)
        w /= w_sum

        for k, patch in enumerate(idx):
            sun_matrix[patch, t] += w[k] * dni[t] / solid_angles[patch]

    return SunResults(matrix=sun_matrix)


def calc_sun_matrix_smooth_smear(sunpath: Sunpath, skydome: Dome, da=15) -> SunResults:
    """
    Smeared sun matrix across multiple patches within smear_angle_deg of the sun direction,
    with stronger weights near the sun and tapering to zero at da.
    """
    sun_vecs = sunpath.sunc.sun_vecs
    dni_vals = sunpath.sunc.dni
    ray_dirs = np.array(skydome.ray_dirs)  # Shape: (N_patches, 3)
    N_patches = len(ray_dirs)
    N_times = len(sun_vecs)
    solid_angles = np.array(skydome.solid_angles)

    sun_matrix = np.zeros((N_patches, N_times))

    smear_angle_rad = np.radians(da)
    total_hits = []
    errors = []

    for i, sun_vec in enumerate(sun_vecs):
        dni = dni_vals[i]
        if dni <= 0.0:
            continue

        sun_vec = sun_vec / np.linalg.norm(sun_vec)  # Safety

        # Compute angles between sun and each patch
        dots = np.clip(np.dot(ray_dirs, sun_vec), -1.0, 1.0)
        angles = np.arccos(dots)  # In radians

        # Identify patches within smear angle
        valid_indices = np.where(angles <= smear_angle_rad)[0]
        total_hits.append(len(valid_indices))

        if len(valid_indices) == 0:
            continue  # fallback to closest patch?

        valid_angles = angles[valid_indices]
        valid_solid_angles = solid_angles[valid_indices]

        # Define a falloff weighting function — e.g., cosine taper
        # w = cos^2(angle / da * pi/2) for smooth falloff to zero
        relative_angle = valid_angles / smear_angle_rad  # Range: 0 to 1
        relative_angle = np.clip(relative_angle, 0, 1)
        weights = np.cos(relative_angle * math.pi / 2) ** 2

        weight_sum = np.sum(weights)
        if weight_sum > 0:
            dni_per_solid_angle = dni * (weights / weight_sum) / valid_solid_angles
            sun_matrix[valid_indices, i] = dni_per_solid_angle
            error = np.abs(np.sum(dni_per_solid_angle * valid_solid_angles) - dni) / dni
            errors.append(error)

    info(f"-----------------------------------------------------")
    info(f"Sun matrix calculation summary (Smooth Smear):")
    info(f"  Average number of patches hit per sun: {np.mean(total_hits):.2f}")
    info(f"  Max patches hit: {np.max(total_hits)}")
    info(f"  Average error in smeared sun matrix: {np.mean(errors):.4f}")
    info(f"-----------------------------------------------------")

    return SunResults(matrix=sun_matrix)


def calc_sun_matrix_from_sunpath(sunpath: Sunpath) -> SunResults:
    T = int(sunpath.sunc.count)
    dni = np.asarray(sunpath.sunc.dni, dtype=np.float32)

    sun_matrix = np.zeros((T, T), dtype=np.float32)
    np.fill_diagonal(sun_matrix, dni)  # or dni>0 filter if you want

    res = SunResults(matrix=sun_matrix)
    res.active_idx = np.arange(T, dtype=np.int32)
    return res


def find_closest_patch(sun_vec, ray_dirs):
    """
    Find the index of the sky patch whose direction is closest to the sun vector.

    Parameters:
    - sun_vec: (3,) unit vector of sun direction
    - ray_dirs: (N, 3) array of unit vectors for sky patches

    Returns:
    - index: int, index of the closest patch
    """
    dots = np.dot(ray_dirs, sun_vec)  # shape (N,)
    return np.argmax(dots)  # max dot = min angle


def patch_occurrences_from_active_idx(active_idx, n_patches: int) -> np.ndarray:
    """
    Count how many times each sundome patch index occurs in active_idx.

    Parameters
    ----------
    active_idx : array-like of int, shape (T,)
        Per-timestep patch indices. Use -1 for "inactive" (sun below horizon).
    n_patches : int
        Number of patches/rays in the sundome (P).

    Returns
    -------
    counts : np.ndarray, shape (n_patches,)
        counts[p] == number of occurrences of patch p in active_idx.
    """
    idx = np.asarray(active_idx, dtype=np.int64).ravel()

    # keep only valid patch indices (ignore -1)
    mask = (idx >= 0) & (idx < n_patches)
    valid = idx[mask]

    # bincount gives counts per integer label; minlength ensures length == n_patches
    counts = np.bincount(valid, minlength=n_patches).astype(np.int32)
    return counts
