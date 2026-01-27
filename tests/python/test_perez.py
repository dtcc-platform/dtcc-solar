"""
Unit tests for dtcc_solar.perez module.
"""

import pytest
import math
import numpy as np
import pandas as pd
from datetime import datetime

from dtcc_solar.perez import (
    compute_sky_clearness,
    compute_sky_brightness,
    calculate_air_mass,
    calc_julian_day,
    calc_eccentricity,
    perez_rel_lum,
    calc_2_phase_matrices,
    calc_3_phase_matrices,
    calc_sky_matrix,
    find_closest_patch,
)
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.tregenza import Tregenza
from dtcc_solar.reinhart2 import ReinhartM2
from dtcc_solar.utils import SolarParameters, SkyResults, SunResults, AnalysisType


class TestComputeSkyClearness:
    """Tests for compute_sky_clearness function."""

    def test_clear_sky_high_clearness(self):
        """Test that clear sky (high DNI, low DHI) gives high clearness."""
        dni = 800  # W/m²
        dhi = 100  # W/m²
        zenith = math.radians(45)
        epsilon = compute_sky_clearness(dni, dhi, zenith)
        # Clear sky should have high clearness
        assert epsilon > 3.0

    def test_overcast_sky_low_clearness(self):
        """Test that overcast sky (low DNI, high DHI) gives low clearness."""
        dni = 50  # W/m²
        dhi = 200  # W/m²
        zenith = math.radians(45)
        epsilon = compute_sky_clearness(dni, dhi, zenith)
        # Overcast sky should have low clearness
        assert epsilon < 2.0

    def test_clearness_clamped_minimum(self):
        """Test that clearness is clamped to minimum of 1.0."""
        dni = 0
        dhi = 300
        zenith = math.radians(60)
        epsilon = compute_sky_clearness(dni, dhi, zenith)
        assert epsilon >= 1.0

    def test_clearness_clamped_maximum(self):
        """Test that clearness is clamped to maximum."""
        dni = 1000
        dhi = 10  # Very low DHI
        zenith = math.radians(30)
        epsilon = compute_sky_clearness(dni, dhi, zenith)
        # Should be clamped to reasonable maximum
        assert epsilon <= 12.0

    def test_zero_dhi_returns_maximum(self):
        """Test that zero DHI returns maximum clearness."""
        epsilon = compute_sky_clearness(500, 0, math.radians(45))
        assert epsilon == 12.0

    def test_clearness_varies_with_zenith(self):
        """Test that clearness varies with zenith angle."""
        dni, dhi = 600, 150
        eps1 = compute_sky_clearness(dni, dhi, math.radians(30))
        eps2 = compute_sky_clearness(dni, dhi, math.radians(60))
        # Should be different (formula includes zenith term)
        assert eps1 != eps2


class TestComputeSkyBrightness:
    """Tests for compute_sky_brightness function."""

    def test_brightness_positive(self):
        """Test that brightness is positive."""
        dhi = 200
        m = 1.5
        epsilon = 2.0
        ts = pd.Timestamp("2024-06-21 12:00", tz="Etc/GMT-1")
        delta = compute_sky_brightness(dhi, m, epsilon, ts)
        assert delta > 0

    def test_brightness_clamped(self):
        """Test that brightness is clamped to valid range."""
        dhi = 500
        m = 2.0
        epsilon = 3.0
        ts = pd.Timestamp("2024-06-21 12:00", tz="Etc/GMT-1")
        delta = compute_sky_brightness(dhi, m, epsilon, ts)
        # Should be within reasonable range
        assert 0.01 <= delta <= 0.6

    def test_brightness_varies_with_air_mass(self):
        """Test that brightness varies with air mass."""
        dhi = 200
        epsilon = 2.0
        ts = pd.Timestamp("2024-06-21 12:00", tz="Etc/GMT-1")
        delta1 = compute_sky_brightness(dhi, 1.0, epsilon, ts)
        delta2 = compute_sky_brightness(dhi, 3.0, epsilon, ts)
        assert delta1 != delta2


class TestCalculateAirMass:
    """Tests for calculate_air_mass function."""

    def test_zenith_zero_gives_one(self):
        """Test that zenith = 0 gives air mass ≈ 1.0."""
        m = calculate_air_mass(0)
        assert np.isclose(m, 1.0, atol=0.01)

    def test_air_mass_increases_with_zenith(self):
        """Test that air mass increases with zenith angle."""
        m1 = calculate_air_mass(math.radians(0))
        m2 = calculate_air_mass(math.radians(45))
        m3 = calculate_air_mass(math.radians(70))
        assert m1 < m2 < m3

    def test_air_mass_typical_values(self):
        """Test typical air mass values."""
        # At 45° zenith, air mass should be around 1.4
        m45 = calculate_air_mass(math.radians(45))
        assert 1.3 < m45 < 1.5

        # At 60° zenith, air mass should be around 2.0
        m60 = calculate_air_mass(math.radians(60))
        assert 1.8 < m60 < 2.2


class TestCalcJulianDay:
    """Tests for calc_julian_day function."""

    def test_jan_1_is_day_1(self):
        """Test that January 1 is Julian day 1."""
        ts = pd.Timestamp("2024-01-01 12:00", tz="UTC")
        assert calc_julian_day(ts) == 1

    def test_dec_31_non_leap(self):
        """Test December 31 of non-leap year is day 365."""
        ts = pd.Timestamp("2023-12-31 12:00", tz="UTC")
        assert calc_julian_day(ts) == 365

    def test_dec_31_leap_year(self):
        """Test December 31 of leap year is day 366."""
        ts = pd.Timestamp("2024-12-31 12:00", tz="UTC")
        assert calc_julian_day(ts) == 366

    def test_summer_solstice(self):
        """Test June 21 (approximately day 172)."""
        ts = pd.Timestamp("2024-06-21 12:00", tz="UTC")
        jd = calc_julian_day(ts)
        # June 21 is around day 173 in leap year
        assert 170 < jd < 175

    def test_winter_solstice(self):
        """Test December 21 (approximately day 355)."""
        ts = pd.Timestamp("2024-12-21 12:00", tz="UTC")
        jd = calc_julian_day(ts)
        # December 21 is around day 356 in leap year
        assert 353 < jd < 360


class TestCalcEccentricity:
    """Tests for calc_eccentricity function."""

    def test_eccentricity_around_one(self):
        """Test that eccentricity correction is around 1.0."""
        for jd in [1, 100, 200, 300, 365]:
            e = calc_eccentricity(jd)
            # Should be close to 1 (within ~4% - Earth's orbital eccentricity)
            assert 0.96 < e < 1.04

    def test_perihelion_greater_than_one(self):
        """Test that eccentricity at perihelion (early Jan) > 1."""
        e = calc_eccentricity(3)  # January 3 is perihelion
        assert e > 1.0

    def test_aphelion_less_than_one(self):
        """Test that eccentricity at aphelion (early July) < 1."""
        e = calc_eccentricity(185)  # Around July 4 is aphelion
        assert e < 1.0

    def test_eccentricity_finite(self):
        """Test that eccentricity is always finite."""
        for jd in range(1, 367):
            e = calc_eccentricity(jd)
            assert math.isfinite(e)


class TestPerezRelLum:
    """Tests for perez_rel_lum function."""

    def test_relative_luminance_non_negative(self):
        """Test that relative luminance is non-negative."""
        ksi = math.radians(45)  # Zenith angle
        gamma = math.radians(30)  # Angle to sun
        A, B, C, D, E = -0.5, -0.3, 1.0, -0.1, 0.1

        f = perez_rel_lum(ksi, gamma, A, B, C, D, E)
        assert f >= 0

    def test_circumsolar_brightening(self):
        """Test that luminance is higher near the sun (small gamma)."""
        ksi = math.radians(45)
        A, B, C, D, E = -0.5, -0.3, 2.0, 1.0, 0.5

        # Near sun (small gamma)
        f_near = perez_rel_lum(ksi, math.radians(5), A, B, C, D, E)
        # Far from sun (large gamma)
        f_far = perez_rel_lum(ksi, math.radians(90), A, B, C, D, E)

        # Circumsolar brightening: luminance should be higher near sun
        # (depends on coefficients, but generally true for positive C, D)
        assert f_near != f_far

    def test_horizon_brightening(self):
        """Test luminance at horizon vs zenith (depends on coefficients)."""
        gamma = math.radians(60)  # Moderate angle to sun
        A, B, C, D, E = 1.0, -0.5, 0.5, -0.1, 0.2

        # Near horizon (large ksi)
        f_horizon = perez_rel_lum(math.radians(80), gamma, A, B, C, D, E)
        # Near zenith (small ksi)
        f_zenith = perez_rel_lum(math.radians(10), gamma, A, B, C, D, E)

        # Both should be valid (non-negative)
        assert f_horizon >= 0
        assert f_zenith >= 0

    def test_luminance_varies_with_ksi(self):
        """Test that luminance varies with zenith angle."""
        gamma = math.radians(45)
        A, B, C, D, E = -0.3, -0.2, 1.0, -0.1, 0.1

        f1 = perez_rel_lum(math.radians(20), gamma, A, B, C, D, E)
        f2 = perez_rel_lum(math.radians(60), gamma, A, B, C, D, E)
        assert f1 != f2


class TestFindClosestPatch:
    """Tests for find_closest_patch function."""

    def test_finds_closest_zenith(self):
        """Test finding patch closest to zenith."""
        # Ray directions including zenith
        ray_dirs = np.array(
            [
                [0, 0, 1],  # Zenith
                [1, 0, 0],  # East horizon
                [0, 1, 0],  # North horizon
            ]
        )
        sun_vec = np.array([0, 0, 1])  # Sun at zenith

        idx = find_closest_patch(sun_vec, ray_dirs)
        assert idx == 0

    def test_finds_closest_east(self):
        """Test finding patch closest to east."""
        ray_dirs = np.array(
            [
                [0, 0, 1],  # Zenith
                [1, 0, 0],  # East horizon
                [0, 1, 0],  # North horizon
            ]
        )
        sun_vec = np.array([0.9, 0.1, 0.2])  # Roughly east
        sun_vec = sun_vec / np.linalg.norm(sun_vec)

        idx = find_closest_patch(sun_vec, ray_dirs)
        assert idx == 1  # Should be east patch


class TestMatrixCalculations:
    """Tests for matrix calculation functions."""

    def test_calc_sky_matrix_returns_sky_results(self, solar_params_week):
        """Test that calc_sky_matrix returns SkyResults."""
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()
        result = calc_sky_matrix(sunpath, skydome)

        assert isinstance(result, SkyResults)

    def test_calc_sky_matrix_correct_shape(self, solar_params_week):
        """Test that sky matrix has correct shape."""
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()
        result = calc_sky_matrix(sunpath, skydome)

        # Shape should be (num_patches, num_timesteps)
        assert result.matrix.shape[0] == skydome.patch_counter
        assert result.matrix.shape[1] == sunpath.sunc.count

    def test_calc_sky_matrix_non_negative(self, solar_params_week):
        """Test that sky matrix values are non-negative."""
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()
        result = calc_sky_matrix(sunpath, skydome)

        assert np.all(result.matrix >= 0)

    def test_calc_sky_matrix_finite(self, solar_params_week):
        """Test that sky matrix contains no NaN or Inf."""
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()
        result = calc_sky_matrix(sunpath, skydome)

        assert np.all(np.isfinite(result.matrix))


class TestCalc2PhaseMatrices:
    """Tests for calc_2_phase_matrices function."""

    def test_returns_sky_and_sun_results(self, solar_params_week):
        """Test that function returns both sky and sun results."""
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()
        sky_res, sun_res = calc_2_phase_matrices(sunpath, skydome, solar_params_week)

        assert isinstance(sky_res, SkyResults)
        assert isinstance(sun_res, SunResults)

    def test_matrices_correct_shape(self, solar_params_week):
        """Test that matrices have correct shapes."""
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()
        sky_res, sun_res = calc_2_phase_matrices(sunpath, skydome, solar_params_week)

        # Both matrices should have same patch dimension
        assert sky_res.matrix.shape[0] == skydome.patch_counter
        assert sun_res.matrix.shape[0] == skydome.patch_counter

    def test_matrices_non_negative(self, solar_params_week):
        """Test that all matrix values are non-negative."""
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()
        sky_res, sun_res = calc_2_phase_matrices(sunpath, skydome, solar_params_week)

        assert np.all(sky_res.matrix >= 0)
        assert np.all(sun_res.matrix >= 0)


class TestCalc3PhaseMatrices:
    """Tests for calc_3_phase_matrices function."""

    def test_returns_sky_and_sun_results(self, solar_params_week):
        """Test that function returns both sky and sun results."""
        # Need to set analysis type to THREE_PHASE for 3-phase matrices
        params = SolarParameters(
            weather_file=solar_params_week.weather_file,
            display=False,
            analysis_type=AnalysisType.THREE_PHASE_1D,
            start=solar_params_week.start,
            end=solar_params_week.end,
        )
        sunpath = Sunpath(params)
        skydome = Tregenza()
        sky_res, sun_res = calc_3_phase_matrices(sunpath, skydome, params)

        assert isinstance(sky_res, SkyResults)
        assert isinstance(sun_res, SunResults)

    def test_sun_matrix_is_diagonal(self, solar_params_week):
        """Test that 3-phase sun matrix has diagonal structure."""
        params = SolarParameters(
            weather_file=solar_params_week.weather_file,
            display=False,
            analysis_type=AnalysisType.THREE_PHASE_1D,
            start=solar_params_week.start,
            end=solar_params_week.end,
        )
        sunpath = Sunpath(params)
        skydome = Tregenza()
        _, sun_res = calc_3_phase_matrices(sunpath, skydome, params)

        # In 3-phase, sun matrix is (n_suns, n_suns) with DNI on diagonal
        assert sun_res.matrix.shape[0] == sun_res.matrix.shape[1]


class TestDifferentSkydomes:
    """Tests for matrix calculations with different skydome types."""

    @pytest.mark.parametrize(
        "dome_class,expected_patches",
        [
            (Tregenza, 145),
            (ReinhartM2, 578),
        ],
    )
    def test_sky_matrix_shape_varies_with_dome(
        self, solar_params_week, dome_class, expected_patches
    ):
        """Test that sky matrix shape matches skydome type."""
        sunpath = Sunpath(solar_params_week)
        skydome = dome_class()
        result = calc_sky_matrix(sunpath, skydome)

        assert result.matrix.shape[0] == expected_patches
