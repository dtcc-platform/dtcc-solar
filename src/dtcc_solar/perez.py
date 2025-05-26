import math
import numpy as np
from dtcc_solar.logging import info, debug, warning, error

perez_coeffs = {
    1: {
        "epsilon_range": (1.000, 1.065),
        "a": -1.000,
        "b": -0.3185,
        "c": 10.1100,
        "d": -3.000,
        "e": 0.450,
    },
    2: {
        "epsilon_range": (1.065, 1.230),
        "a": -0.9585,
        "b": -0.2515,
        "c": 8.6000,
        "d": -1.500,
        "e": 0.3086,
    },
    3: {
        "epsilon_range": (1.230, 1.500),
        "a": -0.7760,
        "b": -0.1623,
        "c": 6.4000,
        "d": -0.750,
        "e": 0.2538,
    },
    4: {
        "epsilon_range": (1.500, 1.950),
        "a": -0.5690,
        "b": -0.0539,
        "c": 3.8000,
        "d": -0.250,
        "e": 0.1774,
    },
    5: {
        "epsilon_range": (1.950, 2.800),
        "a": -0.3277,
        "b": 0.0653,
        "c": 1.5000,
        "d": 0.000,
        "e": 0.1062,
    },
    6: {
        "epsilon_range": (2.800, 4.500),
        "a": -0.2500,
        "b": 0.1500,
        "c": 0.0000,
        "d": 0.000,
        "e": 0.0000,
    },
    7: {
        "epsilon_range": (4.500, 6.200),
        "a": -0.1500,
        "b": 0.3000,
        "c": -1.2500,
        "d": 0.000,
        "e": 0.0000,
    },
    8: {
        "epsilon_range": (6.200, float("inf")),
        "a": -0.1000,
        "b": 0.4000,
        "c": -2.5000,
        "d": 0.000,
        "e": 0.0000,
    },
}


def get_perez_coeffs(epsilon):
    for sky_type, data in perez_coeffs.items():
        lower, upper = data["epsilon_range"]
        if lower <= epsilon <= upper:
            return [data["a"], data["b"], data["c"], data["d"], data["e"]]
        elif epsilon < lower:
            warning("Epsilon value out of range (1.0-inf), is zenith > 90 degrees?")
    return None  # or raise ValueError("Epsilon out of range")


def compute_epsilon(dni, dhi, zenith):
    """
    Compute the Perez clearness index (ε) used for sky classification.

    Parameters:
        dhi (float): Diffuse Horizontal Irradiance [W/m²]
        dni (float): Direct Normal Irradiance [W/m²]
        zenith_rad (float): Solar zenith angle [radians]

    Returns:
        float: Perez clearness index ε
    """

    # Cosine of zenith angle (used to project DNI onto horizontal plane)
    cos_zenith = math.cos(zenith)

    # Clearness index formula (Perez 1990)

    term1 = dhi + dni * cos_zenith
    term2 = dhi + 1.041 * math.pow(zenith, 3)
    epsilon = term1 / term2

    if epsilon < 1.0:
        # warning(f"Epsilon {epsilon:.3f} is below valid range. Clamping to 1.0.")
        epsilon = 1.0

    return epsilon


def per_rel_luminance(sun_zenith, sun_patch_angle, a, b, c, d, e):
    """
    Compute the relative luminance f(θ, γ) of a sky patch using the Perez model.

    Parameters:
        zenith : float
            Zenith angle of the patch (in radians, 0 at zenith, π/2 at horizon)
        sun_patch_angle : float
            Angle between sun vector and patch direction (in radians)
        a, b, c, d, e : float
            Perez sky model coefficients

    Returns:
        float
            Relative luminance (unitless)
    """
    cos_sun_zenith = max(np.cos(sun_zenith), 0.001)  # avoid divide-by-zero
    cos_sun_patch_angle = np.cos(sun_patch_angle)

    term1 = 1 + a * np.exp(b / cos_sun_zenith)
    term2 = 1 + c * np.exp(d * sun_patch_angle) + e * cos_sun_patch_angle**2

    f = term1 * term2

    return f


def calc_norm_factor(dhi, fs, zenith_angles, solid_angles):
    """
    Compute the Perez normalization factor I from precomputed relative luminance.

    Parameters:
        dhi : float
            Diffuse Horizontal Irradiance [W/m²]
        fs : np.ndarray
            Array of unnormalized relative luminance f_i for each patch
        zenith_angles : np.ndarray
            Array of zenith angles θ_i for each patch [radians]
        solid_angles : np.ndarray
            Array of solid angles ΔΩ_i for each patch [steradians]

    Returns:
        float : normalization factor I
    """
    cos_zenith = np.clip(np.cos(zenith_angles), 0.01, 1.0)  # avoid division artifacts
    weights = fs * cos_zenith * solid_angles
    denominator = np.sum(weights)

    if denominator == 0.0:
        return 0.0  # or raise an error if desired

    return dhi / denominator
