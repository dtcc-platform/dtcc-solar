"""
Unit tests for dtcc_solar.coefficients module.
"""

import pytest
import math
import numpy as np

from dtcc_solar.coefficients import (
    clearness_bins,
    perez_coeff_table,
    zenith_lum_coeffs,
    diffuse_lum_eff,
    direct_lum_eff,
    find_bin_index,
    calc_perez_coeffs,
    calc_zenith_lum_coeffs,
    compute_coeff,
    compute_special_c,
    compute_special_d,
)


class TestClearnessBins:
    """Tests for clearness bin boundaries."""

    def test_clearness_bins_count(self):
        """Test that there are 8 clearness bins."""
        assert len(clearness_bins) == 8

    def test_clearness_bins_lower_bound(self):
        """Test that first bin starts at 1.0."""
        assert clearness_bins[0][0] == 1.0

    def test_clearness_bins_upper_bound(self):
        """Test that last bin extends to infinity."""
        assert clearness_bins[-1][1] == float("inf")

    def test_clearness_bins_contiguous(self):
        """Test that bins are contiguous (no gaps)."""
        for i in range(len(clearness_bins) - 1):
            assert clearness_bins[i][1] == clearness_bins[i + 1][0]

    def test_clearness_bins_monotonic(self):
        """Test that bin boundaries are monotonically increasing."""
        prev_upper = 0
        for low, high in clearness_bins:
            assert low >= prev_upper
            assert high > low
            prev_upper = high


class TestPerezCoeffTable:
    """Tests for Perez coefficient table."""

    def test_perez_coeff_table_has_8_rows(self):
        """Test that table has 8 rows (one per bin)."""
        assert len(perez_coeff_table) == 8

    def test_perez_coeff_table_has_20_columns(self):
        """Test that each row has 20 coefficients."""
        for row in perez_coeff_table:
            assert len(row) == 20

    def test_perez_coefficients_are_finite(self):
        """Test that all coefficients are finite numbers."""
        for row in perez_coeff_table:
            for coeff in row:
                assert math.isfinite(coeff)


class TestZenithLumCoeffs:
    """Tests for zenith luminance coefficients."""

    def test_zenith_lum_coeffs_has_8_rows(self):
        """Test that there are 8 sets of coefficients."""
        assert len(zenith_lum_coeffs) == 8

    def test_zenith_lum_coeffs_has_4_values(self):
        """Test that each set has 4 coefficients."""
        for coeffs in zenith_lum_coeffs:
            assert len(coeffs) == 4

    def test_zenith_lum_coeffs_are_finite(self):
        """Test that all coefficients are finite."""
        for coeffs in zenith_lum_coeffs:
            for c in coeffs:
                assert math.isfinite(c)


class TestLuminousEfficacy:
    """Tests for luminous efficacy coefficient tables."""

    def test_diffuse_lum_eff_has_8_rows(self):
        """Test diffuse luminous efficacy has 8 rows."""
        assert len(diffuse_lum_eff) == 8

    def test_diffuse_lum_eff_has_4_values(self):
        """Test each diffuse lum eff row has 4 values."""
        for row in diffuse_lum_eff:
            assert len(row) == 4

    def test_direct_lum_eff_has_8_rows(self):
        """Test direct luminous efficacy has 8 rows."""
        assert len(direct_lum_eff) == 8

    def test_direct_lum_eff_has_4_values(self):
        """Test each direct lum eff row has 4 values."""
        for row in direct_lum_eff:
            assert len(row) == 4


class TestFindBinIndex:
    """Tests for find_bin_index function."""

    def test_bin_index_minimum_value(self):
        """Test bin index for minimum epsilon (1.0)."""
        assert find_bin_index(1.0) == 0

    def test_bin_index_first_bin(self):
        """Test bin index within first bin."""
        assert find_bin_index(1.03) == 0

    def test_bin_index_boundary(self):
        """Test bin index at boundary (should go to next bin)."""
        assert find_bin_index(1.065) == 1

    def test_bin_index_second_bin(self):
        """Test bin index within second bin."""
        assert find_bin_index(1.15) == 1

    def test_bin_index_middle_bins(self):
        """Test bin indices for middle bins."""
        # Bin 2: [1.230, 1.500)
        assert find_bin_index(1.3) == 2
        # Bin 3: [1.500, 1.950)
        assert find_bin_index(1.7) == 3
        # Bin 4: [1.950, 2.800)
        assert find_bin_index(2.0) == 4
        # Bin 5: [2.800, 4.500)
        assert find_bin_index(3.5) == 5
        # Bin 6: [4.500, 6.200)
        assert find_bin_index(5.0) == 6

    def test_bin_index_last_bin(self):
        """Test bin index for high clearness values."""
        assert find_bin_index(6.5) == 7
        assert find_bin_index(10.0) == 7
        assert find_bin_index(100.0) == 7

class TestCalcPerezCoeffs:
    """Tests for calc_perez_coeffs function."""

    def test_returns_5_coefficients(self):
        """Test that function returns 5 coefficients (a, b, c, d, e)."""
        epsilon = 1.5
        delta = 0.1
        zenith_rad = math.radians(45)
        result = calc_perez_coeffs(epsilon, delta, zenith_rad)
        assert len(result) == 5

    def test_coefficients_are_finite(self):
        """Test that all returned coefficients are finite."""
        epsilon = 2.0
        delta = 0.2
        zenith_rad = math.radians(30)
        a, b, c, d, e = calc_perez_coeffs(epsilon, delta, zenith_rad)
        assert math.isfinite(a)
        assert math.isfinite(b)
        assert math.isfinite(c)
        assert math.isfinite(d)
        assert math.isfinite(e)

    def test_coefficients_vary_with_epsilon(self):
        """Test that coefficients change with different epsilon values."""
        delta = 0.15
        zenith_rad = math.radians(45)

        result1 = calc_perez_coeffs(1.2, delta, zenith_rad)
        result2 = calc_perez_coeffs(5.0, delta, zenith_rad)

        # At least some coefficients should differ
        assert result1 != result2

    def test_coefficients_vary_with_delta(self):
        """Test that coefficients change with different delta values."""
        epsilon = 2.0
        zenith_rad = math.radians(45)

        result1 = calc_perez_coeffs(epsilon, 0.05, zenith_rad)
        result2 = calc_perez_coeffs(epsilon, 0.3, zenith_rad)

        # At least some coefficients should differ
        assert result1 != result2

    def test_coefficients_vary_with_zenith(self):
        """Test that coefficients change with different zenith angles."""
        epsilon = 3.0
        delta = 0.2

        result1 = calc_perez_coeffs(epsilon, delta, math.radians(20))
        result2 = calc_perez_coeffs(epsilon, delta, math.radians(70))

        # At least some coefficients should differ
        assert result1 != result2

    def test_first_bin_special_formulas(self):
        """Test coefficients for first bin (uses special formulas for c, d)."""
        epsilon = 1.03  # Should use bin 0 (special formulas)
        delta = 0.1
        zenith_rad = math.radians(45)
        a, b, c, d, e = calc_perez_coeffs(epsilon, delta, zenith_rad)

        # Just verify they're finite - special formulas are complex
        assert math.isfinite(c)
        assert math.isfinite(d)

    def test_typical_clear_sky(self):
        """Test coefficients for typical clear sky conditions."""
        epsilon = 6.0  # High clearness (clear sky)
        delta = 0.1
        zenith_rad = math.radians(45)
        a, b, c, d, e = calc_perez_coeffs(epsilon, delta, zenith_rad)

        # All coefficients should be reasonable (no extreme values)
        for coeff in [a, b, c, d, e]:
            assert abs(coeff) < 1000

    def test_typical_overcast_sky(self):
        """Test coefficients for typical overcast sky conditions."""
        epsilon = 1.1  # Low clearness (overcast)
        delta = 0.3
        zenith_rad = math.radians(60)
        a, b, c, d, e = calc_perez_coeffs(epsilon, delta, zenith_rad)

        # All coefficients should be reasonable
        for coeff in [a, b, c, d, e]:
            assert abs(coeff) < 1000


class TestCalcZenithLumCoeffs:
    """Tests for calc_zenith_lum_coeffs function."""

    def test_returns_4_coefficients(self):
        """Test that function returns 4 coefficients."""
        result = calc_zenith_lum_coeffs(2.0)
        assert len(result) == 4

    def test_coefficients_are_finite(self):
        """Test that all coefficients are finite."""
        result = calc_zenith_lum_coeffs(3.0)
        for c in result:
            assert math.isfinite(c)

    def test_coefficients_vary_with_epsilon(self):
        """Test that coefficients differ for different epsilon bins."""
        result1 = calc_zenith_lum_coeffs(1.0)  # Bin 0
        result2 = calc_zenith_lum_coeffs(5.0)  # Bin 6

        # Should get different coefficient sets
        assert list(result1) != list(result2)

    def test_returns_correct_row(self):
        """Test that correct row is returned for given epsilon."""
        # Epsilon 1.5 should be in bin 3
        result = calc_zenith_lum_coeffs(1.5)
        expected = zenith_lum_coeffs[3]
        assert list(result) == list(expected)


class TestComputeCoeff:
    """Tests for compute_coeff function."""

    def test_compute_coeff_basic(self):
        """Test basic coefficient computation."""
        x1, x2, x3, x4 = 1.0, 2.0, 3.0, 4.0
        Z = math.radians(45)
        delta = 0.1

        result = compute_coeff(x1, x2, x3, x4, Z, delta)

        # Expected: x1 + (x2 * Z) + delta * (x3 + (x4 * Z))
        expected = x1 + (x2 * Z) + delta * (x3 + (x4 * Z))
        assert math.isclose(result, expected)

    def test_compute_coeff_zero_delta(self):
        """Test coefficient computation with zero delta."""
        x1, x2, x3, x4 = 1.0, 2.0, 3.0, 4.0
        Z = math.radians(30)
        delta = 0.0

        result = compute_coeff(x1, x2, x3, x4, Z, delta)
        expected = x1 + (x2 * Z)
        assert math.isclose(result, expected)


class TestComputeSpecialC:
    """Tests for compute_special_c function (bin 0 special formula)."""

    def test_compute_special_c_returns_finite(self):
        """Test that special c formula returns finite value."""
        c1, c2, c3, c4 = 2.8, 0.6004, 1.2375, 1.0
        Z = math.radians(45)
        delta = 0.1

        result = compute_special_c(c1, c2, c3, c4, Z, delta)
        assert math.isfinite(result)

    def test_compute_special_c_small_delta(self):
        """Test special c with small delta value."""
        c1, c2, c3, c4 = 2.8, 0.6004, 1.2375, 1.0
        Z = math.radians(30)
        delta = 0.01

        result = compute_special_c(c1, c2, c3, c4, Z, delta)
        assert math.isfinite(result)


class TestComputeSpecialD:
    """Tests for compute_special_d function (bin 0 special formula)."""

    def test_compute_special_d_returns_finite(self):
        """Test that special d formula returns finite value."""
        d1, d2, d3, d4 = 1.8734, 0.6297, 0.9738, 0.2809
        Z = math.radians(45)
        delta = 0.1

        result = compute_special_d(d1, d2, d3, d4, Z, delta)
        assert math.isfinite(result)

    def test_compute_special_d_clamping(self):
        """Test that exponent clamping prevents overflow."""
        d1, d2, d3, d4 = 10.0, 10.0, 0.0, 0.0
        Z = math.radians(80)  # Large zenith
        delta = 1.0  # Large delta

        # Should not raise overflow error due to clamping
        result = compute_special_d(d1, d2, d3, d4, Z, delta)
        assert math.isfinite(result)
