import math
import numpy as np
import pandas as pd
import pprint as pp
from dtcc_solar.logging import info, debug, warning, error
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.skydome import Skydome
from dtcc_solar.tregenza import Tregenza
from dtcc_solar.reinhart2 import ReinhartM2
from dataclasses import dataclass, field
from dtcc_solar.coefficients import calc_perez_coeffs
from dtcc_solar.plotting import sub_plots_2d, sub_plot_dict, plot_coeffs_dict
from dtcc_solar.plotting import show_plot, plot_debug_1, plot_debug_2
from dtcc_solar.utils import SkyResults, SunResults, SunMapping


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


def calc_2_phase_matrix(
    sunpath: Sunpath, skydome: Skydome, type: SunMapping = SunMapping.RADIANCE
) -> list[SkyResults, SunResults]:
    sky_res = calc_sky_matrix(sunpath, skydome)

    if type == SunMapping.NONE:
        sun_res = calc_sun_matrix(sunpath, skydome)
    elif type == SunMapping.RADIANCE:
        sun_res = calc_sun_matrix_rad(sunpath, skydome)

    calc_tot_error(sky_res, skydome, sun_res, sunpath)

    return (sky_res, sun_res)


def calc_3_phase_matrices(
    sunpath: Sunpath,
    skydome: Skydome,
) -> tuple[SkyResults, SunResults]:
    sky_res = calc_sky_matrix(sunpath, skydome)
    sun_res = calc_sun_matrix_from_sunpath(sunpath)
    calc_tot_error(sky_res, skydome, sun_res, sunpath)

    return sky_res, sun_res


def calc_sky_matrix(sunpath: Sunpath, skydome: Skydome) -> SkyResults:

    dni = sunpath.sunc.dni
    dhi = sunpath.sunc.dhi
    sun_vecs = sunpath.sunc.sun_vecs
    sun_zenith = sunpath.sunc.zeniths
    sun_times = sunpath.sunc.time_stamps

    rel_lum = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])
    nor_lum = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])
    sky_mat = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])

    all_ksis = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])
    all_gammas = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])

    solid_angles = np.array(skydome.solid_angles)

    zenith_limit = math.radians(89.9)  # Limit for zenith angle for numerical stability
    norm_limit = 0.01  # Normalisation factor limit for uniform sky

    small_norms = 0
    eval_count = 0

    ignored_dhi = 0.0

    for i in range(len(sun_vecs)):

        if dhi[i] > 0.0 and sun_zenith[i] < zenith_limit:

            air_mass = calculate_air_mass(sun_zenith[i])
            epsilon = compute_sky_clearness(dni[i], dhi[i], sun_zenith[i])
            delta = compute_sky_brightness(dhi[i], air_mass, epsilon, sun_times[i])

            [A, B, C, D, E] = calc_perez_coeffs(epsilon, delta, sun_zenith[i])

            lvs = []
            ksis = []

            for j in range(len(skydome.ray_dirs)):
                ray_dir = np.array(skydome.ray_dirs[j])
                sun_patch_dot = np.dot(sun_vecs[i], ray_dir)
                gamma = math.acos(sun_patch_dot)
                gamma = min(max(gamma, 1e-4), math.pi)
                ksi = skydome.patch_zeniths[j]

                lv = perez_rel_lum(ksi, gamma, A, B, C, D, E)
                lvs.append(lv)
                ksis.append(ksi)

                all_ksis[j, i] = ksi
                all_gammas[j, i] = gamma

            lvs = np.array(lvs)
            ksis = np.array(ksis)

            # Calculate normalisation factor eq. (3) in Perez 1993
            norm = np.sum(lvs * np.cos(ksis) * solid_angles)

            if norm <= norm_limit:
                # Uniform sky: distribute DHI equally per solid angle
                small_norms += 1
                L = dhi[i] / math.pi
                Rvs = np.full_like(ksis, L)  # (W/m²*sr)
            else:
                # Calculate absolute radiance
                Rvs = (lvs * dhi[i]) / norm  # (W/m²*sr)

            rel_lum[:, i] = lvs
            sky_mat[:, i] = Rvs  #   (W/m²/sr)

            projected_sum = np.sum(Rvs * np.cos(ksis) * solid_angles)  # (W/m²)
            eval_count += 1
        else:
            # If no diffuse radiation, set to zero
            rel_lum[:, i] = 0.0
            sky_mat[:, i] = 0.0
            ignored_dhi += dhi[i]

    n_suns = len(sun_vecs)
    info("-----------------------------------------------------")
    info("Sky matrix calculation summary (Perez):")
    info(f"  Evaluated {eval_count} sun positions of {n_suns} which passed the checks.")
    info(f"  Conditions: dhi > 0 and sun zenith < {math.degrees(zenith_limit)} °")
    info(f"  For {small_norms} cases the norm factor <  {norm_limit} => uniform sky")
    info("-----------------------------------------------------")

    # Store as class attributes
    perez_results = SkyResults()
    perez_results.count = len(sun_vecs)
    perez_results.relative_luminance = rel_lum
    perez_results.solid_angles = solid_angles
    perez_results.matrix = sky_mat
    perez_results.ksis = all_ksis
    perez_results.gammas = all_gammas
    perez_results.ignored_dhi = ignored_dhi

    return perez_results


def calc_tot_error(
    sky_res: SkyResults, skydome: Skydome, sun_res: SunResults, sunp: Sunpath
):

    patch_zeniths = np.array(skydome.patch_zeniths)

    sun_dni = np.sum(np.sum(sun_res.matrix, axis=1))
    sky_dhi = np.sum(np.sum(sky_res.matrix, axis=1) * np.cos(patch_zeniths))

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


def calc_sun_matrix(sunpath: Sunpath, skydome: Skydome) -> SunResults:

    sun_vecs = sunpath.sunc.sun_vecs
    sun_matrix = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])
    ray_dirs = np.array(skydome.ray_dirs)

    for i in range(len(sun_vecs)):
        patch_index = find_closest_patch(sun_vecs[i], ray_dirs)
        patch_solid_angle = skydome.solid_angles[patch_index]
        sun_matrix[patch_index, i] = sunpath.sunc.dni[i] / patch_solid_angle  # W/m²/sr

    sun_results = SunResults()
    sun_results.matrix = sun_matrix

    return sun_results


def calc_sun_matrix_rad(
    sunpath: Sunpath, skydome: Skydome, n_targets: int = 4
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


def calc_sun_matrix_from_sunpath(sunpath: Sunpath) -> SunResults:

    sun_matrix = np.zeros((sunpath.sunc.count, sunpath.sunc.count))
    dni = sunpath.sunc.dni

    for i in range(len(sunpath.sunc.sun_vecs)):
        sun_matrix[i, i] = dni[i]

    sun_res = SunResults()
    sun_res.matrix = sun_matrix

    return sun_res


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
