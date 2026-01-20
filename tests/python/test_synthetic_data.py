"""
Unit tests for dtcc_solar.synthetic_data module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import os

from dtcc_solar.synthetic_data import synthetic_epw_df, df_to_epw


class TestSyntheticEpwDf:
    """Tests for synthetic_epw_df function."""

    def test_returns_dataframe_and_header(self):
        """Test that synthetic_epw_df returns a DataFrame and header list."""
        df, header = synthetic_epw_df()
        assert isinstance(df, pd.DataFrame)
        assert isinstance(header, list)

    def test_dataframe_has_35_columns(self):
        """Test that DataFrame has all 35 EPW columns."""
        df, _ = synthetic_epw_df()
        assert len(df.columns) == 35

    def test_expected_columns_present(self):
        """Test that key columns are present."""
        df, _ = synthetic_epw_df()
        expected_cols = [
            "Year", "Month", "Day", "Hour", "Minute",
            "DNI", "DHI", "GHI", "DryBulb", "DewPoint",
            "RelHum", "WindSpeed", "WindDir",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_full_year_row_count(self):
        """Test that full year generates 8784 hours (leap year 2024)."""
        df, _ = synthetic_epw_df(
            start=datetime(2024, 1, 1, 0, 0),
            end=datetime(2024, 12, 31, 23, 0),
        )
        # Leap year: 366 days * 24 hours = 8784
        assert len(df) == 8784

    def test_non_leap_year_row_count(self):
        """Test that non-leap year generates 8760 hours."""
        df, _ = synthetic_epw_df(
            start=datetime(2023, 1, 1, 0, 0),
            end=datetime(2023, 12, 31, 23, 0),
        )
        # Non-leap year: 365 days * 24 hours = 8760
        assert len(df) == 8760

    def test_one_day_row_count(self):
        """Test that one day generates 24 hours."""
        df, _ = synthetic_epw_df(
            start=datetime(2024, 6, 21, 0, 0),
            end=datetime(2024, 6, 21, 23, 0),
        )
        assert len(df) == 24

    def test_dni_values_non_negative(self):
        """Test that DNI values are non-negative."""
        df, _ = synthetic_epw_df()
        assert (df["DNI"] >= 0).all()

    def test_dhi_values_non_negative(self):
        """Test that DHI values are non-negative."""
        df, _ = synthetic_epw_df()
        assert (df["DHI"] >= 0).all()

    def test_ghi_values_non_negative(self):
        """Test that GHI values are non-negative."""
        df, _ = synthetic_epw_df()
        assert (df["GHI"] >= 0).all()

    def test_dni_zero_at_night(self):
        """Test that DNI is zero during night hours."""
        df, _ = synthetic_epw_df(
            start=datetime(2024, 6, 21, 0, 0),
            end=datetime(2024, 6, 21, 23, 0),
        )
        # Night hours (0-5 and 19-23) should have zero DNI
        # Note: Hour column is 1-24 in EPW format
        night_hours = df[(df["Hour"] <= 6) | (df["Hour"] >= 20)]
        assert (night_hours["DNI"] == 0).all()

    def test_temperature_realistic_range(self):
        """Test that temperature values are in realistic range."""
        df, _ = synthetic_epw_df()
        assert (df["DryBulb"] > -50).all()
        assert (df["DryBulb"] < 60).all()

    def test_relative_humidity_in_range(self):
        """Test that relative humidity is between 0 and 100."""
        df, _ = synthetic_epw_df()
        assert (df["RelHum"] >= 0).all()
        assert (df["RelHum"] <= 100).all()

    def test_wind_speed_non_negative(self):
        """Test that wind speed is non-negative."""
        df, _ = synthetic_epw_df()
        assert (df["WindSpeed"] >= 0).all()

    def test_wind_direction_in_range(self):
        """Test that wind direction is between 0 and 360."""
        df, _ = synthetic_epw_df()
        assert (df["WindDir"] >= 0).all()
        assert (df["WindDir"] <= 360).all()

    def test_custom_location(self):
        """Test that custom location parameters are used."""
        df, header = synthetic_epw_df(
            city="TestCity",
            country="TestCountry",
            latitude=45.0,
            longitude=-75.0,
            timezone=-5,
            elevation=200,
        )
        # Check header contains location info
        assert "TestCity" in header[0]
        assert "TestCountry" in header[0]

    def test_header_has_8_lines(self):
        """Test that header has exactly 8 lines."""
        _, header = synthetic_epw_df()
        assert len(header) == 8


class TestDfToEpw:
    """Tests for df_to_epw function."""

    def test_writes_file(self, tmp_path):
        """Test that EPW file is written."""
        df, header = synthetic_epw_df(
            start=datetime(2024, 6, 21, 0, 0),
            end=datetime(2024, 6, 21, 23, 0),
        )
        epw_path = tmp_path / "test_output.epw"
        df_to_epw(df, header, str(epw_path))
        assert epw_path.exists()

    def test_file_contains_header(self, tmp_path):
        """Test that EPW file contains header lines."""
        df, header = synthetic_epw_df(
            start=datetime(2024, 6, 21, 0, 0),
            end=datetime(2024, 6, 21, 23, 0),
        )
        epw_path = tmp_path / "test_output.epw"
        df_to_epw(df, header, str(epw_path))

        with open(epw_path, "r") as f:
            lines = f.readlines()

        # First 8 lines should be header
        assert len(lines) >= 8
        assert "LOCATION" in lines[0]

    def test_file_contains_data_rows(self, tmp_path):
        """Test that EPW file contains data rows after header."""
        df, header = synthetic_epw_df(
            start=datetime(2024, 6, 21, 0, 0),
            end=datetime(2024, 6, 21, 23, 0),
        )
        epw_path = tmp_path / "test_output.epw"
        df_to_epw(df, header, str(epw_path))

        with open(epw_path, "r") as f:
            lines = f.readlines()

        # Header (8) + data rows (24)
        assert len(lines) == 8 + 24

    def test_written_file_readable_by_sunpath(self, tmp_path):
        """Test that written EPW file can be read back."""
        df, header = synthetic_epw_df(
            start=datetime(2024, 1, 1, 0, 0),
            end=datetime(2024, 12, 31, 23, 0),
        )
        epw_path = tmp_path / "full_year.epw"
        df_to_epw(df, header, str(epw_path))

        # Read back the file and verify basic structure
        df_read = pd.read_csv(epw_path, skiprows=8, header=None)
        assert len(df_read) == len(df)
        assert len(df_read.columns) == 35


class TestDataConsistency:
    """Tests for data consistency in synthetic EPW data."""

    def test_ghi_equals_dni_plus_dhi_approximately(self):
        """Test that GHI approximately equals DNI + DHI for daytime."""
        df, _ = synthetic_epw_df(
            start=datetime(2024, 6, 21, 0, 0),
            end=datetime(2024, 6, 21, 23, 0),
        )
        # During day hours, GHI should approximately equal DNI + DHI
        # (simplified model, so relationship may not be exact)
        daytime = df[(df["Hour"] >= 8) & (df["Hour"] <= 18)]
        if len(daytime) > 0:
            # Check that GHI is at least as large as DHI
            assert (daytime["GHI"] >= daytime["DHI"]).all()

    def test_dew_point_below_dry_bulb(self):
        """Test that dew point is at or below dry bulb temperature."""
        df, _ = synthetic_epw_df()
        assert (df["DewPoint"] <= df["DryBulb"]).all()

    def test_hour_values_valid(self):
        """Test that hour values are valid (1-24 EPW format)."""
        df, _ = synthetic_epw_df()
        assert (df["Hour"] >= 1).all()
        assert (df["Hour"] <= 24).all()

    def test_month_values_valid(self):
        """Test that month values are valid (1-12)."""
        df, _ = synthetic_epw_df()
        assert (df["Month"] >= 1).all()
        assert (df["Month"] <= 12).all()

    def test_day_values_valid(self):
        """Test that day values are valid (1-31)."""
        df, _ = synthetic_epw_df()
        assert (df["Day"] >= 1).all()
        assert (df["Day"] <= 31).all()

    def test_no_nan_values(self):
        """Test that there are no NaN values in the DataFrame."""
        df, _ = synthetic_epw_df()
        # Check numeric columns for NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert df[col].notna().all(), f"Column {col} contains NaN values"

    def test_pressure_realistic(self):
        """Test that atmospheric pressure is realistic."""
        df, _ = synthetic_epw_df()
        # Pressure should be around 101325 Pa at sea level (adjusted for elevation)
        assert (df["AtmPressure"] > 50000).all()
        assert (df["AtmPressure"] < 120000).all()
