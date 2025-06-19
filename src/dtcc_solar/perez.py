import math
import numpy as np
import pandas as pd
import pprint as pp
from dtcc_solar.logging import info, debug, warning, error
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.skydome import Skydome
from dataclasses import dataclass, field
from dtcc_solar.coefficients import calc_perez_coeffs
from dtcc_solar.plotting import sub_plots_2d, sub_plot_dict, plot_coeffs_dict
from dtcc_solar.plotting import show_plot, plot_debug_1, plot_debug_2
from dtcc_solar.utils import SkyResults, SunResults, SunMatrixType


"""
Perez sky luminance distribution model implementation.
This module implements the Perez model for sky luminance distribution,
which is used to calculate the luminance of different patches of the sky
based on the position of the sun and sky conditions.
It includes functions to compute sky clearness, zenith luminance,
relative luminance, and absolute luminance for sky patches.
It also provides a data class to store the results of the calculations.
"""


def compute_sky_clearness(dni, dhi, sun_zenith_rad):
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


def calculate_air_mass(sun_zenith):
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


def compute_sky_brightness(dhi, m, epsilon, ts: pd.Timestamp, ext=1367):
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


def calc_sky_sun_matrix(
    sunpath: Sunpath,
    skydome: Skydome,
    type: SunMatrixType = SunMatrixType.smooth_smear,
    da: float = 15.0,
):
    sky_res = calc_sky_matrix(sunpath, skydome)

    if type == SunMatrixType.straight:
        sun_res = calc_sun_matrix(sunpath, skydome)
    elif type == SunMatrixType.flat_smear:
        sun_res = calc_sun_mat_flat_smear(sunpath, skydome, da)
    elif type == SunMatrixType.smooth_smear:
        sun_res = calc_sun_mat_smooth_smear(sunpath, skydome, da)

    calc_tot_error(sky_res, skydome, sun_res, sunpath)

    tot_matrix = sky_res.sky_matrix + sun_res.sun_matrix

    return tot_matrix


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

    keys_1 = ["sun_zenith", "dni", "dhi", "epsilon", "delta", "air_mass"]
    data_dict_1 = {key: [] for key in keys_1}

    keys_2 = ["Patch_irr_max", "Patch_irr_min", "norm", "error_dhi", "error_lum_norm"]
    data_dict_2 = {key: [] for key in keys_2}

    coeffs_keys = ["A", "B", "C", "D", "E"]
    coeffs_dict = {key: [] for key in coeffs_keys}

    solid_angles = np.array(skydome.solid_angles)

    zenith_limit = math.radians(89.9)  # Limit for zenith angle for numerical stability
    norm_limit = 0.01  # Normalisation factor limit for uniform sky

    small_norm_count = 0
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
                small_norm_count += 1
                weight = np.cos(ksis) * np.sum(solid_angles)  # (sr)
                R_uniform = dhi[i] / weight  # (W/m²·sr)
                Rvs = np.full_like(ksis, R_uniform)
                lvs_norm = np.full_like(ksis, 1.0 / len(lvs))
            else:
                # Calculate absolute radiance
                Rvs = (lvs * dhi[i]) / norm  # (W/m²*sr)
                lvs_norm = lvs / np.sum(lvs)  # Normalised relative luminance (unitless)

            rel_lum[:, i] = lvs
            nor_lum[:, i] = lvs_norm  # Normalised luminance (W/m²·sr)
            sky_mat[:, i] = Rvs * solid_angles  # (W/m²)

            projected_sum = np.sum(Rvs * np.cos(ksis) * solid_angles)  # (W/m²)
            error_dhi = (projected_sum - dhi[i]) / dhi[i]  # (unitless)

            error_lum_norm = np.sum(lvs_norm) - 1.0

            coeffs_dict["A"].append(A)
            coeffs_dict["B"].append(B)
            coeffs_dict["C"].append(C)
            coeffs_dict["D"].append(D)
            coeffs_dict["E"].append(E)

            data_dict_1["sun_zenith"].append(sun_zenith[i])
            data_dict_1["dni"].append(dni[i])
            data_dict_1["dhi"].append(dhi[i])
            data_dict_1["epsilon"].append(epsilon)
            data_dict_1["delta"].append(delta)
            data_dict_1["air_mass"].append(air_mass)

            data_dict_2["Patch_irr_max"].append(np.max(Rvs))
            data_dict_2["Patch_irr_min"].append(np.min(Rvs))
            data_dict_2["norm"].append(norm)
            data_dict_2["error_dhi"].append(error_dhi)
            data_dict_2["error_lum_norm"].append(error_lum_norm)

            eval_count += 1

        else:
            # If no diffuse radiation, set to zero
            rel_lum[:, i] = 0.0
            sky_mat[:, i] = 0.0
            ignored_dhi += dhi[i]

    info(f"Evaluated {eval_count} sun positions out of {len(sun_vecs)}.")
    info(f"Number of norm factors smaller then {norm_limit}: {small_norm_count}")

    # Store as class attributes
    perez_results = SkyResults()
    perez_results.count = len(sun_vecs)
    perez_results.relative_luminance = rel_lum
    perez_results.relative_lum_norm = nor_lum
    perez_results.solid_angles = solid_angles
    perez_results.sky_matrix = sky_mat
    perez_results.ksis = all_ksis
    perez_results.gammas = all_gammas
    perez_results.ignored_dhi = ignored_dhi

    # sub_plot_dict(data_dict_1, 0, 5000)
    # sub_plot_dict(data_dict_2, 0, 5000)
    # plot_coeffs_dict(coeffs_dict, 0, len(sun_vecs), title="Perez Coefficients")
    # plot_debug_1(np.array(debug_gammas), np.array(debug_rvs))
    # plot_debug_2(np.array(debug_sun_zen), np.array(debug_max_rvs))
    # show_plot()

    return perez_results


def calc_tot_error(
    sky_res: SkyResults, skydome: Skydome, sun_res: SunResults, sunp: Sunpath
):

    patch_zeniths = np.array(skydome.patch_zeniths)

    sun_dni = np.sum(np.sum(sun_res.sun_matrix, axis=1))
    sky_dhi = np.sum(np.sum(sky_res.sky_matrix, axis=1) * np.cos(patch_zeniths))

    sum_dni = np.sum(sunp.sunc.dni)
    sum_dhi = np.sum(sunp.sunc.dhi)

    if sum_dni > 0:
        error_dni = np.abs(sun_dni - sum_dni) / sum_dni

    if sum_dhi > 0:
        error_dhi = np.abs(sky_dhi - sum_dhi) / sum_dhi
        error_ign = sky_res.ignored_dhi / sum_dhi

    info(f"Total error in DNI: {100 * error_dni:.3f} %")
    info(f"Total error in DHI: {100 * error_dhi:.3f} %")
    info(f"Total ignored DHI: {100 * error_ign:.3f} %")


def calc_sun_matrix(sunpath: Sunpath, skydome: Skydome) -> SunResults:

    sun_vecs = sunpath.sunc.sun_vecs
    sun_matrix = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])
    ray_dirs = np.array(skydome.ray_dirs)

    for i in range(len(sun_vecs)):
        patch_index = find_closest_patch(sun_vecs[i], ray_dirs)
        sun_matrix[patch_index, i] = sunpath.sunc.dni[i]

    sun_results = SunResults()
    sun_results.sun_matrix = sun_matrix

    return sun_results


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


def calc_sun_mat_flat_smear(sunpath: Sunpath, skydome: Skydome, da=15.0) -> SunResults:
    """
    Smeared sun matrix across multiple patches within smear_angle_deg of the sun direction.
    """
    sun_vecs = sunpath.sunc.sun_vecs
    dni_vals = sunpath.sunc.dni
    ray_dirs = np.array(skydome.ray_dirs)  # Ensure shape (N, 3)
    N_patches = len(ray_dirs)
    N_times = len(sun_vecs)

    sun_matrix = np.zeros((N_patches, N_times))

    dot_thresh = np.cos(np.radians(da))
    total_hits = []

    for i, sun_vec in enumerate(sun_vecs):
        if dni_vals[i] <= 0.0:
            continue

        # Normalize sun vector just in case
        sun_vec = sun_vec / np.linalg.norm(sun_vec)

        # Dot products with all ray directions
        dots = np.dot(ray_dirs, sun_vec)
        valid_indices = np.where(dots >= dot_thresh)[0]
        total_hits.append(len(valid_indices))

        if len(valid_indices) == 0:
            continue  # fallback to single patch?

        sun_matrix[valid_indices, i] = dni_vals[i] / len(valid_indices)

    print(f"Average number of patches hit per sun: {np.mean(total_hits):.2f}")
    print(f"Max patches hit: {np.max(total_hits)}")

    return SunResults(sun_matrix=sun_matrix)


def calc_sun_mat_smooth_smear(sunpath: Sunpath, skydome: Skydome, da=15) -> SunResults:
    """
    Smeared sun matrix across multiple patches within smear_angle_deg of the sun direction,
    with stronger weights near the sun and tapering to zero at da.
    """
    sun_vecs = sunpath.sunc.sun_vecs
    dni_vals = sunpath.sunc.dni
    ray_dirs = np.array(skydome.ray_dirs)  # Shape: (N_patches, 3)
    N_patches = len(ray_dirs)
    N_times = len(sun_vecs)

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

        # Define a falloff weighting function — e.g., cosine taper
        # w = cos^2(angle / da * pi/2) for smooth falloff to zero
        relative_angle = valid_angles / smear_angle_rad  # Range: 0 to 1
        weights = np.cos(relative_angle * math.pi / 2) ** 2

        weight_sum = np.sum(weights)
        if weight_sum > 0:
            dni_contribution = dni * (weights / weight_sum)
            sun_matrix[valid_indices, i] = dni_contribution
            error = np.abs(np.sum(dni_contribution) - dni) / dni
            errors.append(error)

    info(f"Average number of patches hit per sun: {np.mean(total_hits):.2f}")
    info(f"Max patches hit: {np.max(total_hits)}")
    info(f"Average error in smeared sun matrix: {np.mean(errors):.4f}")

    return SunResults(sun_matrix=sun_matrix)


def avg_patch_spacing(skydome: Skydome) -> float:
    n = len(skydome.ray_dirs)
    angles = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = np.clip(np.dot(skydome.ray_dirs[i], skydome.ray_dirs[j]), -1.0, 1.0)
            angle = np.arccos(dot)
            angles.append(np.degrees(angle))
    return np.mean(angles), np.min(angles), np.max(angles)
