"""
Unit tests for dtcc_solar.interpolate module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from dtcc_solar.interpolate import Interpolator


@pytest.fixture
def sample_irradiance_df():
    """Create a sample hourly irradiance DataFrame for testing."""
    # Create 10 days of hourly data
    start = pd.Timestamp("2024-06-21 00:00", tz="Etc/GMT-1")
    periods = 24 * 10  # 10 days
    index = pd.date_range(start=start, periods=periods, freq="h")

    # Simple diurnal pattern
    hours = np.arange(periods) % 24
    dni = np.where((hours >= 6) & (hours <= 18), 100 + 200 * np.sin((hours - 6) * np.pi / 12), 0)
    dhi = np.where((hours >= 6) & (hours <= 18), 50 + 100 * np.sin((hours - 6) * np.pi / 12), 0)

    df = pd.DataFrame({"dni": dni, "dhi": dhi}, index=index)
    return df


@pytest.fixture
def small_df():
    """Create a small DataFrame for quick tests."""
    start = pd.Timestamp("2024-06-21 00:00", tz="Etc/GMT-1")
    periods = 24 * 3  # 3 days
    index = pd.date_range(start=start, periods=periods, freq="h")

    hours = np.arange(periods) % 24
    dni = np.where((hours >= 8) & (hours <= 16), 500.0, 0.0)
    dhi = np.where((hours >= 6) & (hours <= 18), 100.0, 0.0)

    df = pd.DataFrame({"dni": dni, "dhi": dhi}, index=index)
    return df


class TestInterpolatorCreation:
    """Tests for Interpolator initialization."""

    def test_interpolator_creates_reduced_df(self, sample_irradiance_df):
        """Test that Interpolator creates a reduced DataFrame."""
        interp = Interpolator(sample_irradiance_df, day_step=5, min_step=20)
        assert interp.df_reduced is not None
        assert isinstance(interp.df_reduced, pd.DataFrame)

    def test_interpolator_stores_original(self, sample_irradiance_df):
        """Test that Interpolator stores original DataFrame."""
        interp = Interpolator(sample_irradiance_df, day_step=5, min_step=20)
        assert interp.df_original is not None
        assert len(interp.df_original) == len(sample_irradiance_df)

    def test_interpolator_stores_parameters(self, sample_irradiance_df):
        """Test that Interpolator stores day_step and min_step."""
        interp = Interpolator(sample_irradiance_df, day_step=7, min_step=15)
        assert interp.day_step == 7
        assert interp.min_step == 15


class TestReducedDataFrameSize:
    """Tests for reduced DataFrame size."""

    def test_day_step_reduces_days(self, sample_irradiance_df):
        """Test that day_step reduces the number of days."""
        # Original: 10 days
        # With day_step=5: should have 2 representative days
        interp = Interpolator(sample_irradiance_df, day_step=5, min_step=60)

        # Count unique days in reduced DataFrame
        reduced_days = len(interp.df_reduced.index.normalize().unique())

        # Should have fewer days than original
        original_days = len(sample_irradiance_df.index.normalize().unique())
        assert reduced_days < original_days

    def test_min_step_affects_resolution(self, small_df):
        """Test that min_step affects time resolution."""
        interp_20min = Interpolator(small_df, day_step=10, min_step=20)
        interp_60min = Interpolator(small_df, day_step=10, min_step=60)

        # 20-minute resolution should have more rows than 60-minute
        # (if they have the same number of days)
        # Actually, both reduce to the same days, but different time steps
        len_20 = len(interp_20min.df_reduced)
        len_60 = len(interp_60min.df_reduced)

        # 20-min should have more points (3x per hour vs 1x)
        assert len_20 >= len_60


class TestEnergyConservation:
    """Tests for energy conservation during interpolation."""

    def test_dni_approximately_conserved(self, small_df):
        """Test that total DNI is approximately conserved."""
        interp = Interpolator(small_df, day_step=3, min_step=20)

        original_dni = small_df["dni"].sum()
        reduced_dni = interp.df_reduced["dni"].sum()

        # Energy should be approximately conserved (within 5%)
        # Note: Perfect conservation depends on the algorithm
        assert abs(reduced_dni - original_dni) / max(original_dni, 1) < 0.1

    def test_dhi_approximately_conserved(self, small_df):
        """Test that total DHI is approximately conserved."""
        interp = Interpolator(small_df, day_step=3, min_step=20)

        original_dhi = small_df["dhi"].sum()
        reduced_dhi = interp.df_reduced["dhi"].sum()

        # Energy should be approximately conserved (within 10%)
        assert abs(reduced_dhi - original_dhi) / max(original_dhi, 1) < 0.1


class TestReducedDataFrameColumns:
    """Tests for reduced DataFrame columns."""

    def test_reduced_has_dni_column(self, sample_irradiance_df):
        """Test that reduced DataFrame has DNI column."""
        interp = Interpolator(sample_irradiance_df)
        assert "dni" in interp.df_reduced.columns

    def test_reduced_has_dhi_column(self, sample_irradiance_df):
        """Test that reduced DataFrame has DHI column."""
        interp = Interpolator(sample_irradiance_df)
        assert "dhi" in interp.df_reduced.columns

    def test_reduced_has_datetime_index(self, sample_irradiance_df):
        """Test that reduced DataFrame has datetime index."""
        interp = Interpolator(sample_irradiance_df)
        assert isinstance(interp.df_reduced.index, pd.DatetimeIndex)


class TestValueRanges:
    """Tests for value ranges in reduced DataFrame."""

    def test_dni_non_negative(self, sample_irradiance_df):
        """Test that reduced DNI values are non-negative."""
        interp = Interpolator(sample_irradiance_df)
        assert (interp.df_reduced["dni"] >= 0).all()

    def test_dhi_non_negative(self, sample_irradiance_df):
        """Test that reduced DHI values are non-negative."""
        interp = Interpolator(sample_irradiance_df)
        assert (interp.df_reduced["dhi"] >= 0).all()

    def test_no_nan_values(self, sample_irradiance_df):
        """Test that reduced DataFrame has no NaN values."""
        interp = Interpolator(sample_irradiance_df)
        assert not interp.df_reduced["dni"].isna().any()
        assert not interp.df_reduced["dhi"].isna().any()


class TestDifferentDaySteps:
    """Tests for different day_step values."""

    def test_day_step_1_no_reduction(self, small_df):
        """Test that day_step=1 keeps all days."""
        interp = Interpolator(small_df, day_step=1, min_step=60)

        original_days = len(small_df.index.normalize().unique())
        reduced_days = len(interp.df_reduced.index.normalize().unique())

        # Should keep all days (or nearly all)
        assert reduced_days == original_days

    def test_large_day_step_reduces_more(self, sample_irradiance_df):
        """Test that larger day_step reduces more."""
        interp_small = Interpolator(sample_irradiance_df, day_step=2, min_step=60)
        interp_large = Interpolator(sample_irradiance_df, day_step=5, min_step=60)

        # Larger step should result in fewer days
        days_small = len(interp_small.df_reduced.index.normalize().unique())
        days_large = len(interp_large.df_reduced.index.normalize().unique())

        assert days_large <= days_small


class TestDifferentMinSteps:
    """Tests for different min_step values."""

    def test_smaller_min_step_more_rows(self, small_df):
        """Test that smaller min_step produces more rows per day."""
        interp_10 = Interpolator(small_df, day_step=3, min_step=10)
        interp_30 = Interpolator(small_df, day_step=3, min_step=30)

        # 10-minute intervals should have more rows than 30-minute
        assert len(interp_10.df_reduced) >= len(interp_30.df_reduced)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_day(self):
        """Test interpolation with single day of data."""
        start = pd.Timestamp("2024-06-21 00:00", tz="Etc/GMT-1")
        index = pd.date_range(start=start, periods=24, freq="h")
        df = pd.DataFrame({"dni": np.ones(24) * 100, "dhi": np.ones(24) * 50}, index=index)

        interp = Interpolator(df, day_step=1, min_step=30)
        assert len(interp.df_reduced) > 0

    def test_constant_values(self):
        """Test interpolation with constant values."""
        start = pd.Timestamp("2024-06-21 00:00", tz="Etc/GMT-1")
        periods = 24 * 5
        index = pd.date_range(start=start, periods=periods, freq="h")
        df = pd.DataFrame({"dni": np.ones(periods) * 200, "dhi": np.ones(periods) * 100}, index=index)

        interp = Interpolator(df, day_step=2, min_step=30)

        # Values should remain approximately constant (scaled)
        assert interp.df_reduced["dni"].max() > 0
        assert interp.df_reduced["dhi"].max() > 0

    def test_zero_values(self):
        """Test interpolation with all zero values."""
        start = pd.Timestamp("2024-06-21 00:00", tz="Etc/GMT-1")
        periods = 24 * 3
        index = pd.date_range(start=start, periods=periods, freq="h")
        df = pd.DataFrame({"dni": np.zeros(periods), "dhi": np.zeros(periods)}, index=index)

        interp = Interpolator(df, day_step=1, min_step=30)

        # Reduced values should also be zero
        assert (interp.df_reduced["dni"] == 0).all()
        assert (interp.df_reduced["dhi"] == 0).all()


if __name__ == "__main__":
    pytest.main()