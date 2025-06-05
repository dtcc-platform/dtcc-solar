from scipy.interpolate import interp1d


def get_perez_coefficients_interpolated(epsilon):
    """
    Interpolates Perez coefficients (A, B, C, D, E) based on sky clearness ε.
    More refined version using interpolation for better accuracy.

    Input:
        epsilon: Sky clearness ε (unitless)
    """
    epsilon_bins = [1.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2, float("inf")]
    A_vals = [-1.0, -0.95, -0.9, -0.8, -0.6, -0.3, 0.0, 0.3]
    B_vals = [-0.32, -0.29, -0.26, -0.23, -0.15, -0.06, 0.00, 0.05]
    C_vals = [10.0, 9.8, 9.5, 9.0, 8.0, 6.5, 5.0, 3.0]
    D_vals = [-3.0, -2.9, -2.8, -2.6, -2.3, -1.7, -1.0, -0.5]
    E_vals = [0.45, 0.43, 0.41, 0.38, 0.34, 0.27, 0.20, 0.10]

    # Clip epsilon to max range (float('inf') not usable here)
    max_eps = epsilon_bins[-2]
    eps_clipped = min(epsilon, max_eps)

    def safe_interp(x, y):
        return interp1d(x[:-1], y, kind="linear", fill_value="extrapolate")(eps_clipped)

    A = float(safe_interp(epsilon_bins, A_vals))
    B = float(safe_interp(epsilon_bins, B_vals))
    C = float(safe_interp(epsilon_bins, C_vals))
    D = float(safe_interp(epsilon_bins, D_vals))
    E = float(safe_interp(epsilon_bins, E_vals))

    return A, B, C, D, E


def get_perez_coeffs(epsilon):
    """
    Retrieve Perez model coefficients (A, B, C, D, E) based on sky clearness ε.
    Simplified version using predefined bins.

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
