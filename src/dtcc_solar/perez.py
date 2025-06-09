import math
import numpy as np
import pandas as pd
from dtcc_solar.logging import info, debug, warning, error
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.skydome import Skydome
from dataclasses import dataclass, field
from dtcc_solar.perez_complete_coeffs import calc_perez_coeffs, calc_zenith_lum_coeffs
import matplotlib.pyplot as plt

"""
Perez sky luminance distribution model implementation.
This module implements the Perez model for sky luminance distribution,
which is used to calculate the luminance of different patches of the sky
based on the position of the sun and sky conditions.
It includes functions to compute sky clearness, zenith luminance,
relative luminance, and absolute luminance for sky patches.
It also provides a data class to store the results of the calculations.
"""


@dataclass
class PerezResults:
    # Number of suns
    count: int = 0
    # 2D array of absolute luminance * solid angle (W/m2) per patch and timestep [n x t]
    sky_vector_matrix: np.ndarray = field(default_factory=lambda: np.empty(0))
    # 2D array of relative luminance F (unitless) per patch and timestep [n x t]
    relative_luminance: np.ndarray = field(default_factory=lambda: np.empty(0))
    # 2D array of norm relative luminance F (unitless) per patch and timestep [n x t]
    relative_lum_norm: np.ndarray = field(default_factory=lambda: np.empty(0))
    # 2D array of absolute luminance L (W/m2/sr) per patch and timestep [n x t]
    absolute_luminance: np.ndarray = field(default_factory=lambda: np.empty(0))
    # 1D array of solid angles (steradians) for each patch [n]
    solid_angles: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Relative luminance term1 (unitless) for Perez model
    rel_lum_term1: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Relative luminance term2 (unitless) for Perez model
    rel_lum_term2: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Ksi angles (zenith angles in radians) for each patch and timestep
    ksis: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Gamma angles (angle between sun and patch in radians) for each patch and timestep
    gammas: np.ndarray = field(default_factory=lambda: np.empty(0))


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


def compute_zenith_luminance_1(dhi, sun_zenith):
    """
    Compute zenith luminance Yz using Perez Option A formula.

    Parameters:
    - dhi: Diffuse horizontal irradiance (W/m²)
    - sun_zenith_rad: Solar zenith angle in radians

    Returns:
    - Yz: Zenith luminance in cd/m²
    """
    term = 0.91 + 10 * math.exp(-3 * sun_zenith) + 0.45 * math.cos(sun_zenith) ** 2
    Yz = (1000 / math.pi) * dhi * term
    return Yz


def compute_zenith_luminance_2(dhi, sun_zenith):
    """
    Compute zenith luminance Yz in W/m²/sr using Perez Option A.
    """
    term = 0.91 + 10 * math.exp(-3 * sun_zenith) + 0.45 * math.cos(sun_zenith) ** 2
    Yz = dhi / math.pi * term
    return Yz


def calc_zenith_luminance_3(epsilon, delta, dhi, Z) -> float:
    """
    Calculate zenith luminance using eqiation (10) in Perez 1990.

    Parameters:
        dhi: Diffuse horizontal irradiance (W/m²)
        Z: Solar zenith angle (radians)
        epsilon: Sky clearness ε (unitless)
        delta: Sky brightness Δ (unitless)
    """

    [ai, ci, cip, di] = calc_zenith_lum_coeffs(epsilon)

    Lvz = dhi * (ai + ci * math.cos(Z) + cip * math.exp(-3 * Z) + di * delta)

    return Lvz


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
    first_day = pd.Timestamp(ts.year, 1, 1)
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


def lux_to_irradiance(lux, efficacy=112):
    """
    Convert illuminance (lux) to irradiance (W/m²).

    Parameters:
    - lux: Illuminance in lux (lm/m²)
    - efficacy: Luminous efficacy in lumens per watt (lm/W), default for daylight = 112

    Returns:
    - Irradiance in W/m²
    """
    return lux / efficacy


def normalize_relative_luminance(Fs, weights=None, target=1.0):
    """
    Normalize the relative luminance values so they sum to a target total.

    Inputs:
        Fs           : ndarray of relative luminance values (unitless)
        weights      : optional solid angle weights for each patch (same shape as l_rel)
        target       : desired sum after normalization

    Returns:
        Normalized relative luminance ndarray.
    """
    if weights is None:
        weights = np.ones_like(Fs)

    weighted_sum = np.sum(Fs * weights)
    if weighted_sum <= 0:
        raise ValueError("Sum of weighted luminance must be positive.")

    scale = target / weighted_sum
    return Fs * scale


def check_azimuthal_symmetry(Fs, patch_zenith, patch_azimuth, sun_azimuth):
    """
    Check symmetry of sky relative luminance with respect to sun azimuth.

    Inputs:
        F           : array of luminance values (same shape as theta, phi)
        theta       : array of patch zenith angles (radians)
        phi         : array of azimuth angles (radians)
        sun_azimuth : sun azimuth angle (radians)

    Returns:
        mean_abs_diff : mean absolute luminance difference between mirror patches
        max_abs_diff  : maximum absolute difference
        summary       : text summary
    """
    phi_mirror = (2 * sun_azimuth - patch_azimuth) % (2 * np.pi)

    # Find indices of closest match for mirror directions
    def find_mirror_indices(phi_array, phi_target):
        return np.argmin(np.abs(phi_array[:, None] - phi_target[None, :]), axis=0)

    mirror_idx = find_mirror_indices(patch_azimuth, phi_mirror)

    l_mirror = Fs[mirror_idx]
    diff = np.abs(Fs - l_mirror)

    mean_diff = np.mean(diff)
    max_diff = np.max(diff)

    summary = f"Symmetry check:\n  Mean abs diff: {mean_diff:.3f}\n  Max abs diff: {max_diff:.3f}"
    return mean_diff, max_diff, summary


def calc_norm_factor(lvs, dhi, ksis, solid_angles):
    """
    Calculate normalisation factor to scale relative luminance values.

    Inputs:
        lvs          : Relative luminance values (unitless)
        ksis         : Zenith angles for each patch (radians)
        solid_angles : Solid angles for each patch (steradians)

    Returns:
        Absolute radiance in W/m²/sr
    """
    integrand = lvs * np.cos(ksis) * solid_angles
    norm_factor = np.sum(integrand)

    warning("Normalization factor is zero or negative, returning zeros.")

    return norm_factor


def calc_sky_matrix(sunpath: Sunpath, skydome: Skydome) -> PerezResults:

    dni = sunpath.sunc.dni
    dhi = sunpath.sunc.dhi
    dni_synth = sunpath.sunc.synth_dni
    dhi_synth = sunpath.sunc.synth_dhi
    sun_vecs = sunpath.sunc.sun_vecs
    sun_zenith = sunpath.sunc.zeniths
    sun_times = sunpath.sunc.time_stamps

    pre_rel_lum = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])
    sky_vec_mat = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])

    all_ksis = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])
    all_gammas = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])
    coeffs = np.zeros([5, len(sun_vecs)])

    keys_1 = ["sun_zenith", "dni", "dhi", "epsilon", "delta", "air_mass"]
    data_dict_1 = {key: [] for key in keys_1}

    keys_2 = ["Rvs_min", "Rvs_max", "normalisation"]
    data_dict_2 = {key: [] for key in keys_2}

    solid_angles = np.array(skydome.solid_angles)

    zenith_limit = math.radians(90)  # Limit for zenith angle for numerical stability

    for i in range(len(sun_vecs)):
        # For each sun position compuite the radiation on each patch
        dni[i] = dni_synth[i]
        dhi[i] = dhi_synth[i]

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
            norm = calc_norm_factor(lvs, dhi[i], ksis, solid_angles)

            if norm <= 1e-7:
                # Uniform sky the ensures dhi is distributed evenly across patches
                cos_weighted_sum = np.sum(np.cos(ksis) * solid_angles)
                R_uniform = dhi[i] / cos_weighted_sum
                Rvs = R_uniform
            else:
                # Calculate absolute radiance
                Rvs = (lvs * dhi[i]) / norm

            pre_rel_lum[:, i] = lvs
            sky_vec_mat[:, i] = Rvs

            coeffs[0, i] = A
            coeffs[1, i] = B
            coeffs[2, i] = C
            coeffs[3, i] = D
            coeffs[4, i] = E

            data_dict_1["sun_zenith"].append(sun_zenith[i])
            data_dict_1["dni"].append(dni[i])
            data_dict_1["dhi"].append(dhi[i])
            data_dict_1["epsilon"].append(epsilon)
            data_dict_1["delta"].append(delta)
            data_dict_1["air_mass"].append(air_mass)

            data_dict_2["Rvs_min"].append(np.min(Rvs))
            data_dict_2["Rvs_max"].append(np.max(Rvs))
            data_dict_2["normalisation"].append(norm)

        else:
            # If no diffuse radiation, set to zero
            pre_rel_lum[:, i] = 0.0
            sky_vec_mat[:, i] = 0.0

    # Store as class attributes
    perez_results = PerezResults()

    perez_results.count = len(sun_vecs)
    perez_results.relative_luminance = pre_rel_lum
    perez_results.sky_vector_matrix = sky_vec_mat
    perez_results.ksis = all_ksis
    perez_results.gammas = all_gammas
    perez_results.solid_angles = solid_angles

    sub_plots_2d(perez_results, 0, 5000)
    sub_plot_dict(data_dict_1, 0, 5000)
    sub_plot_dict(data_dict_2, 0, 5000)
    plot_coeffs_data(coeffs, "coeffs", 0, len(sun_vecs), title="Perez Coefficients")

    plt.show()

    return perez_results


def sub_plots_2d(res: PerezResults, t1, t2):

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    axs = axs.flatten()  # Flatten to simplify indexing

    subplot2d(axs[0], res.relative_luminance, t1, t2, title="Relative Luminance")
    subplot2d(axs[1], res.sky_vector_matrix, t1, t2, title="Sky Vector Matrix")

    plt.tight_layout()


def sub_plot_dict(data: dict, t1, t2):

    n = len(data.keys())
    fig, axs = plt.subplots(nrows=n, ncols=1, figsize=(16, 10))
    axs = axs.flatten()  # Flatten to simplify indexing
    counter = 0

    t2 = min(t2, len(next(iter(data.values()))))

    for key, value in data.items():
        subplot1d(axs[counter], value, t1, t2, name=key)
        counter += 1

    plt.tight_layout()


def subplot2d(ax, data: np.ndarray, t1=0, t2=5, title="title"):
    """
    Plot a slice of 2D data on a given Axes subplot.
    """
    n_patches, n_timesteps = data.shape
    upper = min(t2, n_timesteps)
    timestep_indices = list(range(t1, upper))

    for i in timestep_indices:
        F_values = data[:, i]
        ax.plot(range(n_patches), F_values, label=f"Sun #{i}")

    ax.set_xlabel("Sky Patch Index")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def subplot1d(ax, data: np.ndarray, t1, t2, name="title"):
    """
    Plot a slice of 2D data on a given Axes subplot.
    """
    data_cropped = data[t1:t2]
    n = len(data_cropped)
    ax.plot(range(t1, t2), data_cropped, label=name)

    ax.set_xlabel("Sun index")
    ax.set_ylabel("Value")
    ax.set_title(name)
    ax.grid(True)
    ax.legend()


def plot_2d_data(data: np.ndarray, t1=0, t2=5, title="title"):

    n_patches, n_timesteps = data.shape
    upper = min(t2, n_timesteps)
    timestep_indices = list(range(t1, upper))

    print(f"Plotting {len(timestep_indices)} timesteps from {t1} to {upper}.")

    plt.figure(figsize=(12, 6))

    for i in timestep_indices:
        F_values = data[:, i]
        plt.plot(range(n_patches), F_values, label=f"Sun #{i}")

    plt.xlabel("Sky Patch Index")
    plt.ylabel("Relative Luminance F")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_coeffs_data(data: np.ndarray, data_name: str, t1=0, t2=5, title="title"):
    """
    Plot a slice of 1D data array.
    """
    n_coeffs, n_suns = data.shape

    coeffs_names = ["A", "B", "C", "D", "E"]

    plt.figure("Perez Coefficients", figsize=(12, 6))

    for i in range(n_coeffs):
        data_cropped = data[i, t1:t2]
        plt.plot(range(t1, t2), data_cropped, label=coeffs_names[i])

    plt.xlabel(f"Sun indices from {t1} to {t2}")
    plt.ylabel(data_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
