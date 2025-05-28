import math
import numpy as np
from dtcc_solar.logging import info, debug, warning, error
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.skydome import Skydome
from dataclasses import dataclass, field


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


def get_perez_coefficients(epsilon):
    """
    Retrieve Perez model coefficients (A, B, C, D, E) based on sky clearness ε
    Input:
        epsilon: Sky clearness ε (unitless)
    Returns:
        Tuple (A, B, C, D, E)
    """
    # Table from Perez et al. (1990)
    perez_bins = [
        (1.065, -1.0, -0.32, 10.0, -3.0, 0.45),
        (1.230, -0.95, -0.29, 9.8, -2.9, 0.43),
        (1.500, -0.90, -0.26, 9.5, -2.8, 0.41),
        (1.950, -0.80, -0.23, 9.0, -2.6, 0.38),
        (2.800, -0.60, -0.15, 8.0, -2.3, 0.34),
        (4.500, -0.30, -0.06, 6.5, -1.7, 0.27),
        (6.200, 0.00, 0.00, 5.0, -1.0, 0.20),
        (float("inf"), 0.30, 0.05, 3.0, -0.5, 0.10),
    ]

    for threshold, A, B, C, D, E in perez_bins:
        if epsilon <= threshold:
            return A, B, C, D, E

    raise ValueError("Clearness value is out of expected range.")


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


def calculate_air_mass(zenith_deg):
    """
    Calculates the relative air mass using Kasten and Young's formula.
    Input:
        zenith_deg: solar zenith angle in degrees
    Returns:
        air mass (unitless)
    """
    if zenith_deg >= 90:
        return float("inf")  # sun below horizon
    z = zenith_deg
    return 1 / (math.cos(math.radians(z)) + 0.50572 * (96.07995 - z) ** -1.6364)


def compute_sky_brightness(dhi, zenith_deg, ext=1367):
    """
    Computes the sky brightness Δ.
    Inputs:
        dhi: Diffuse horizontal irradiance (W/m²)
        zenith_deg: Solar zenith angle in degrees
        ext: Extraterrestrial solar irradiance (W/m²), default 1367
    Returns:
        Brightness Δ (unitless)
    """
    m = calculate_air_mass(zenith_deg)
    return (dhi * m) / ext


def perez_rel_lum(patch_zenith, sun_patch_angle, A, B, C, D, E):
    """
    Computes the Perez relative luminance distribution function F(θ, γ).

    Inputs:
        patch_zenith: Zenith angle θ for sky patch (radians)
        sun_patch_angle: Angle between sky patch and sun (radians)
        A-E: Perez coefficients
    Returns:
        Relative luminance factor F (unitless)
    """
    cos_patch_zenith = math.cos(patch_zenith)
    if cos_patch_zenith == 0:
        cos_patch_zenith = 1e-6  # avoid division by zero

    term1 = 1 + A * math.exp(B / cos_patch_zenith)
    term2 = 1 + C * math.exp(D * sun_patch_angle) + E * math.cos(sun_patch_angle) ** 2
    return term1 * term2


def perez_rel_lum_zenith(sun_zenith, A, B, C, D, E):
    """
    Computes the relative luminance Fz(θ, γ) for the zenith patch.

    Inputs:
        sun_zenith: Zenith angle for the sun (radians)
        A-E: Perez coefficients
    Returns:
        Relative luminance at zenith (unitless)
    """
    term1 = 1 + A * np.exp(B)
    term2 = 1 + C * np.exp(D * sun_zenith) + E * (np.cos(sun_zenith) ** 2)
    return term1 * term2


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


def calc_sky_matrix(sunpath: Sunpath, skydome: Skydome) -> PerezResults:

    dni = sunpath.sunc.dni
    dhi = sunpath.sunc.dhi
    sun_vecs = sunpath.sunc.sun_vecs
    sun_zenith = sunpath.sunc.zeniths

    pre_rel_lum = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])
    absolute_lum = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])
    sky_vec_mat = np.zeros([len(skydome.ray_dirs), len(sun_vecs)])

    solid_angles = np.array(skydome.solid_angles)

    for i in range(len(sun_vecs)):
        # For each sun position compuite the radiation on each patch
        if dhi[i] > 0.0 and sun_zenith[i] < np.pi / 2.0:
            epsilon = compute_sky_clearness(dni[i], dhi[i], sun_zenith[i])
            [A, B, C, D, E] = get_perez_coefficients(epsilon)

            Fs = []  # List to store the relative luminance for each patch
            # Compute the radiation for each ray (i.e. each sky patch)
            for j in range(len(skydome.ray_dirs)):
                ray_dir = np.array(skydome.ray_dirs[j])
                sun_patch_dot = np.dot(sun_vecs[i], ray_dir)
                sun_patch_angle = math.acos(sun_patch_dot)

                F = perez_rel_lum(sun_zenith[i], sun_patch_angle, A, B, C, D, E)
                Fs.append(F)

            Fs = np.array(Fs)

            # Compute relative luminance for patch perpendicular to the sun vector
            Fnorm = perez_rel_lum_zenith(sun_zenith[i], A, B, C, D, E)

            # Compute the zenith luminance in cd/m²
            # Yz = compute_zenith_luminance_1(dhi[i], sun_zenith[i])

            # Compute the zenith luminance in W/m²/sr
            Yz = compute_zenith_luminance_2(dhi[i], sun_zenith[i])

            # Absolute luminance for patch i
            Li = Yz * (Fs / Fnorm)

            pre_rel_lum[:, i] = Fs
            absolute_lum[:, i] = Li
            sky_vec_mat[:, i] = Li * solid_angles

            # Normalize relative luminance so that it sums to 1.0
            Fs_norm = normalize_relative_luminance(Fs, target=1.0)

        else:
            # If no diffuse radiation, set to zero
            pre_rel_lum[:, i] = 0.0
            absolute_lum[:, i] = 0.0
            sky_vec_mat[:, i] = 0.0

    # Store as class attributes

    perez_results = PerezResults()

    perez_results.count = len(sun_vecs)
    perez_results.relative_luminance = pre_rel_lum
    perez_results.absolute_luminance = absolute_lum
    perez_results.sky_vector_matrix = sky_vec_mat
    perez_results.solid_angles = solid_angles

    return perez_results
