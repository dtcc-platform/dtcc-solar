import math
from typing import Tuple

# Sky clearness bins and their boundaries from Perez 1990/1993.
clearness_bins = [
    (1.000, 1.065),
    (1.065, 1.230),
    (1.230, 1.500),
    (1.500, 1.950),
    (1.950, 2.800),
    (2.800, 4.500),
    (4.500, 6.200),
    (6.200, float("inf")),
]

# Perez table coefficients for each bin: a1–e4 from table 1 in Perez 1993.
# [a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4, e1, e2, e3, e4]
perez_coeff_table = [
    [
        1.3525,
        -0.2576,
        -0.2690,
        -1.4366,
        -0.7670,
        0.0007,
        1.2734,
        -0.1233,
        2.8000,
        0.6004,
        1.2375,
        1.0000,
        1.8734,
        0.6297,
        0.9738,
        0.2809,
        0.0356,
        -0.1246,
        -0.5718,
        0.9938,
    ],
    [
        -1.2219,
        -0.7730,
        1.4148,
        1.1016,
        -0.2054,
        0.0367,
        -3.9128,
        0.9156,
        6.9750,
        0.1774,
        6.4477,
        -0.1239,
        -1.5798,
        -0.5081,
        -1.7812,
        0.1080,
        0.2624,
        0.0672,
        -0.2190,
        -0.4285,
    ],
    [
        -1.1000,
        -0.2515,
        0.8952,
        0.0156,
        0.2782,
        -0.1812,
        -4.5000,
        1.1766,
        24.7219,
        -13.0812,
        -37.7000,
        34.8438,
        -5.0000,
        1.5218,
        3.9229,
        -2.6204,
        -0.0156,
        0.1597,
        0.4199,
        -0.5562,
    ],
    [
        -0.5484,
        -0.6654,
        -0.2672,
        0.7117,
        0.7234,
        -0.6219,
        -5.6812,
        2.6297,
        33.3389,
        -18.3000,
        -62.2500,
        52.0781,
        -3.5000,
        0.0016,
        1.1477,
        0.1062,
        0.4659,
        -0.3296,
        -0.0876,
        -0.0329,
    ],
    [
        -0.6000,
        -0.3566,
        -2.5000,
        2.3250,
        0.2937,
        0.0496,
        -5.6812,
        1.8415,
        21.0000,
        -4.7656,
        -21.5906,
        7.2492,
        -3.5000,
        -0.1554,
        1.4062,
        0.3988,
        0.0032,
        0.0766,
        -0.0656,
        -0.1294,
    ],
    [
        -1.0156,
        -0.3670,
        1.0078,
        1.4051,
        0.2875,
        -0.5328,
        -3.8500,
        3.3750,
        14.0000,
        -0.9999,
        -7.1406,
        7.5469,
        -3.4000,
        -0.1078,
        -1.0750,
        1.5702,
        -0.0672,
        0.4016,
        0.3017,
        -0.4844,
    ],
    [
        -1.0000,
        0.0211,
        0.5025,
        -0.5119,
        -0.3000,
        0.1922,
        0.7023,
        -1.6317,
        19.0000,
        -5.0000,
        1.2438,
        -1.9094,
        -4.0000,
        0.0250,
        0.3844,
        0.2656,
        1.0468,
        -0.3788,
        -2.4517,
        1.4656,
    ],
    [
        -1.0500,
        0.0289,
        0.4260,
        0.3590,
        -0.3250,
        0.1156,
        0.7781,
        0.0025,
        31.0625,
        -14.5000,
        -46.1148,
        55.3750,
        -7.2312,
        0.4050,
        13.3500,
        0.6234,
        1.5000,
        -0.6426,
        1.8564,
        0.5636,
    ],
]


# Zenith luminance prediction for Eq.(10) in Perez 1990.
zenith_lum_coeffs = [
    [40.86, 26.77, -29.59, -45.75],
    [26.58, 14.73, 58.46, -21.25],
    [19.34, 2.28, 100.0, 0.25],
    [13.25, -1.39, 124.79, 15.66],
    [14.47, -5.09, 160.09, 9.13],
    [19.76, -3.88, 154.61, -19.21],
    [28.39, -9.67, 151.58, -69.39],
    [42.91, -19.62, 130.80, -164.08],
]


# Diffuse luminous efficacy model coefficients (Table 4, Eqn. 7) in Perez 1990.
diffuse_lum_eff = [
    [97.24, -0.46, 12.00, -8.91],
    [107.22, 1.15, 0.59, -3.95],
    [104.97, 2.96, -5.53, -8.77],
    [102.39, 5.59, -13.95, -13.90],
    [100.71, 5.94, -22.75, -23.74],
    [106.42, 3.83, -36.15, -28.83],
    [141.88, 1.90, -53.24, -14.03],
    [152.23, 0.35, -45.27, -7.98],
]

# Direct luminous efficacy model coefficients (Table 4, Eqn. 8) in Perez 1990.
direct_lum_eff = [
    [57.20, -4.55, -2.98, 117.12],
    [98.99, -3.46, -1.21, 12.38],
    [109.83, -4.90, -1.71, -8.81],
    [110.34, -5.84, -1.99, -4.56],
    [106.36, -3.97, -1.75, -6.16],
    [107.19, -1.25, -1.51, -26.73],
    [105.75, 0.77, -1.26, -34.44],
    [101.18, 1.58, -1.10, -8.29],
]


def find_bin_index(epsilon: float) -> int:
    for i, (low, high) in enumerate(clearness_bins):
        if low <= epsilon < high:
            return i
    last_bin = len(clearness_bins) - 1

    return last_bin


def compute_coeff(x1, x2, x3, x4, Z, delta):
    """
    Compute Perez model coefficient for a given parameter. Eq. (6) in the Perez 1993.

    Parameters:
    - x1, x2, x3, x4: Coefficients from the Perez table
    - Z: Solar zenith angle in radians
    - delta: Sky brightness Δ (unitless)

    Returns:
    - Resulting Perez coefficient (float)
    """
    return x1 + (x2 * Z) + delta * (x3 + (x4 * Z))


def compute_special_c(c1, c2, c3, c4, Z, delta):
    """
    Compute coefficient c for clearness bin 1 using corrected formula:
    c = exp([delta * (c1 + c2 * Z)]^c3) - c4, eq.(7) in Perez 1993.

    Parameters:
        c1, c2, c3, c4 : float - Empirical Perez coefficients
        Z : float - Solar zenith angle in radians
        delta : float - Sky brightness (unitless)

    Returns:
        c : float
    """
    base = max(delta * (c1 + c2 * Z), 1e-6)
    c = math.exp(math.pow(base, c3)) - c4

    return c


def compute_special_d(d1, d2, d3, d4, Z, delta):
    """
    Compute coefficient d for clearness bin 1 using corrected formula:
    d = -exp[delta * (d1 + d2 * Z)] + d3 + delta * d4. Eq. (8) in Perez 1993.

    Parameters:
        d1, d2, d3, d4 : float - Empirical Perez coefficients
        Z : float - Solar zenith angle in radians
        delta : float - Sky brightness (unitless)

    Returns:
        d : float
    """
    exponent = delta * (d1 + d2 * Z)
    exponent = min(exponent, 50)  # Prevent overflow
    d = -math.exp(exponent) + d3 + delta * d4
    return d


def calc_perez_coeffs(epsilon: float, delta: float, sun_zenith_rad: float) -> Tuple:

    i = find_bin_index(epsilon)
    Z = sun_zenith_rad
    row = perez_coeff_table[i]

    if i == 0:
        c = compute_special_c(row[8], row[9], row[10], row[11], Z, delta)
        d = compute_special_d(row[12], row[13], row[14], row[15], Z, delta)
    else:
        c = compute_coeff(row[8], row[9], row[10], row[11], Z, delta)
        d = compute_coeff(row[12], row[13], row[14], row[15], Z, delta)

    a = compute_coeff(row[0], row[1], row[2], row[3], Z, delta)
    b = compute_coeff(row[4], row[5], row[6], row[7], Z, delta)
    e = compute_coeff(row[16], row[17], row[18], row[19], Z, delta)

    return a, b, c, d, e


def calc_zenith_lum_coeffs(epsilon: float) -> list[float]:
    """
    Calculate zenith luminance coefficients based on sky clearness ε.

    Parameters:
        epsilon: Sky clearness ε (unitless)

    Returns:
        List of coefficients (ai, ci, cip, di)
    """
    i = find_bin_index(epsilon)

    return zenith_lum_coeffs[i]
