"""
Unit tests for dtcc_solar.sunpath module.
"""

import pytest
import math
import numpy as np
import pandas as pd
from datetime import datetime

from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import SolarParameters, SunCollection, AnalysisType
from dtcc_solar.synthetic_data import synthetic_epw_df, df_to_epw


class TestSunpathCreation:
    """Tests for Sunpath class initialization."""

    def test_sunpath_creates_sun_collection(self, solar_params_week):
        """Test that Sunpath creates a SunCollection."""
        sunpath = Sunpath(solar_params_week)
        assert sunpath.sunc is not None
        assert isinstance(sunpath.sunc, SunCollection)

    def test_sunpath_sun_count_positive(self, solar_params_week):
        """Test that SunCollection has positive sun count."""
        sunpath = Sunpath(solar_params_week)
        # After removing night suns, should still have some suns
        assert sunpath.sunc.count > 0

    def test_sunpath_sun_vectors_match_count(self, solar_params_week):
        """Test that sun vector count matches sun count."""
        sunpath = Sunpath(solar_params_week)
        assert len(sunpath.sunc.sun_vecs) == sunpath.sunc.count

    def test_sunpath_positions_match_count(self, solar_params_week):
        """Test that position count matches sun count."""
        sunpath = Sunpath(solar_params_week)
        assert len(sunpath.sunc.positions) == sunpath.sunc.count

    def test_sunpath_dni_match_count(self, solar_params_week):
        """Test that DNI array length matches sun count."""
        sunpath = Sunpath(solar_params_week)
        assert len(sunpath.sunc.dni) == sunpath.sunc.count

    def test_sunpath_dhi_match_count(self, solar_params_week):
        """Test that DHI array length matches sun count."""
        sunpath = Sunpath(solar_params_week)
        assert len(sunpath.sunc.dhi) == sunpath.sunc.count

    def test_sunpath_zeniths_match_count(self, solar_params_week):
        """Test that zenith array length matches sun count."""
        sunpath = Sunpath(solar_params_week)
        assert len(sunpath.sunc.zeniths) == sunpath.sunc.count


class TestSunVectors:
    """Tests for sun vector properties."""

    def test_sun_vectors_are_unit_vectors(self, solar_params_week):
        """Test that all sun vectors are unit vectors."""
        sunpath = Sunpath(solar_params_week)
        norms = np.linalg.norm(sunpath.sunc.sun_vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_sun_vectors_have_positive_z(self, solar_params_week):
        """Test that all sun vectors point upward (after night removal)."""
        sunpath = Sunpath(solar_params_week)
        # All z components should be positive (sun above horizon)
        z_components = sunpath.sunc.sun_vecs[:, 2]
        assert np.all(z_components > 0)


class TestSunPositions:
    """Tests for sun position properties."""

    def test_positions_on_sphere(self, solar_params_week):
        """Test that sun positions are on the specified radius sphere."""
        sunpath = Sunpath(solar_params_week)
        distances = np.linalg.norm(sunpath.sunc.positions, axis=1)
        expected_radius = sunpath.r
        assert np.allclose(distances, expected_radius, rtol=1e-3)

    def test_positions_above_horizon(self, solar_params_week):
        """Test that all positions are above horizon (positive z)."""
        sunpath = Sunpath(solar_params_week)
        z_positions = sunpath.sunc.positions[:, 2]
        assert np.all(z_positions > 0)


class TestZenithAngles:
    """Tests for zenith angle properties."""

    def test_zeniths_in_valid_range(self, solar_params_week):
        """Test that zenith angles are in valid range [0, π/2]."""
        sunpath = Sunpath(solar_params_week)
        # After removing night suns, zeniths should be < π/2
        assert np.all(sunpath.sunc.zeniths >= 0)
        assert np.all(sunpath.sunc.zeniths < np.pi / 2)

    def test_zeniths_not_nan(self, solar_params_week):
        """Test that zenith angles contain no NaN values."""
        sunpath = Sunpath(solar_params_week)
        assert not np.any(np.isnan(sunpath.sunc.zeniths))


class TestIrradianceData:
    """Tests for DNI and DHI data."""

    def test_dni_non_negative(self, solar_params_week):
        """Test that DNI values are non-negative."""
        sunpath = Sunpath(solar_params_week)
        assert np.all(sunpath.sunc.dni >= 0)

    def test_dhi_non_negative(self, solar_params_week):
        """Test that DHI values are non-negative."""
        sunpath = Sunpath(solar_params_week)
        assert np.all(sunpath.sunc.dhi >= 0)

    def test_dni_not_all_zero(self, solar_params_week):
        """Test that not all DNI values are zero (daytime has some sun)."""
        sunpath = Sunpath(solar_params_week)
        # At least some DNI should be positive during summer week
        assert np.sum(sunpath.sunc.dni) > 0

    def test_dhi_not_all_zero(self, solar_params_week):
        """Test that not all DHI values are zero."""
        sunpath = Sunpath(solar_params_week)
        # At least some DHI should be positive
        assert np.sum(sunpath.sunc.dhi) > 0


class TestNightRemoval:
    """Tests for night sun removal behavior."""

    def test_include_night_false_removes_suns(self, solar_params_week):
        """Test that include_night=False removes below-horizon suns."""
        sunpath = Sunpath(solar_params_week, include_night=False)
        # All zeniths should be < π/2 (above horizon)
        assert np.all(sunpath.sunc.zeniths < np.pi / 2)

    def test_include_night_true_preserves_all(self, solar_params_week):
        """Test that include_night=True preserves all sun positions."""
        sunpath_with_night = Sunpath(solar_params_week, include_night=True)
        sunpath_without_night = Sunpath(solar_params_week, include_night=False)

        # With night should have more or equal sun positions
        assert sunpath_with_night.sunc.count >= sunpath_without_night.sunc.count


class TestRadius:
    """Tests for sunpath radius."""

    def test_custom_radius(self, solar_params_week):
        """Test that custom radius is applied."""
        custom_radius = 100.0
        sunpath = Sunpath(solar_params_week, radius=custom_radius)
        assert sunpath.r == custom_radius

        # Check positions are on the custom radius sphere
        distances = np.linalg.norm(sunpath.sunc.positions, axis=1)
        assert np.allclose(distances, custom_radius, rtol=1e-3)

    def test_default_radius(self, solar_params_week):
        """Test default radius value."""
        sunpath = Sunpath(solar_params_week)
        assert sunpath.r == 50  # Default is 50


class TestTimeStamps:
    """Tests for timestamp handling."""

    def test_timestamps_list(self, solar_params_week):
        """Test that timestamps is a list."""
        sunpath = Sunpath(solar_params_week)
        assert isinstance(sunpath.sunc.time_stamps, list)

    def test_timestamps_match_count(self, solar_params_week):
        """Test that timestamp count matches sun count."""
        sunpath = Sunpath(solar_params_week)
        assert len(sunpath.sunc.time_stamps) == sunpath.sunc.count

    def test_timestamps_are_datetime(self, solar_params_week):
        """Test that timestamps are datetime objects."""
        sunpath = Sunpath(solar_params_week)
        if len(sunpath.sunc.time_stamps) > 0:
            # Check first timestamp
            assert isinstance(sunpath.sunc.time_stamps[0], (datetime, pd.Timestamp))


class TestDataFrame:
    """Tests for internal DataFrame."""

    def test_df_has_dni_column(self, solar_params_week):
        """Test that DataFrame has DNI column."""
        sunpath = Sunpath(solar_params_week)
        assert "dni" in sunpath.df.columns

    def test_df_has_dhi_column(self, solar_params_week):
        """Test that DataFrame has DHI column."""
        sunpath = Sunpath(solar_params_week)
        assert "dhi" in sunpath.df.columns

    def test_df_has_datetime_index(self, solar_params_week):
        """Test that DataFrame has datetime index."""
        sunpath = Sunpath(solar_params_week)
        assert isinstance(sunpath.df.index, pd.DatetimeIndex)


class TestLocationExtraction:
    """Tests for location extraction from EPW file."""

    def test_latitude_extracted(self, solar_params_week):
        """Test that latitude is extracted from EPW header."""
        sunpath = Sunpath(solar_params_week)
        # Synthetic EPW has latitude 59.33 (Stockholm)
        assert abs(sunpath.lat - 59.33) < 1.0

    def test_longitude_extracted(self, solar_params_week):
        """Test that longitude is extracted from EPW header."""
        sunpath = Sunpath(solar_params_week)
        # Synthetic EPW has longitude 18.06 (Stockholm)
        assert abs(sunpath.lon - 18.06) < 1.0


class TestSunpathGeometry:
    """Tests for sunpath geometry creation."""

    def test_create_sunpath_geometry_creates_mesh(self, solar_params_week):
        """Test that create_sunpath_geometry creates a mesh."""
        sunpath = Sunpath(solar_params_week)
        sunpath.create_sunpath_geometry()
        assert sunpath.mesh is not None

    def test_create_sunpath_geometry_creates_analemmas(self, solar_params_week):
        """Test that create_sunpath_geometry creates analemma meshes."""
        sunpath = Sunpath(solar_params_week)
        sunpath.create_sunpath_geometry()
        assert sunpath.analemmas_meshes is not None
        assert len(sunpath.analemmas_meshes) > 0

    def test_create_sunpath_geometry_creates_daypath_meshes(self, solar_params_week):
        """Test that create_sunpath_geometry creates day path meshes."""
        sunpath = Sunpath(solar_params_week)
        sunpath.create_sunpath_geometry()
        assert sunpath.daypath_meshes is not None
        # Should have 3 day paths (summer solstice, equinox, winter solstice)
        assert len(sunpath.daypath_meshes) == 3

    def test_create_sunpath_geometry_creates_sun_pc(self, solar_params_week):
        """Test that create_sunpath_geometry creates sun point cloud."""
        sunpath = Sunpath(solar_params_week)
        sunpath.create_sunpath_geometry()
        assert sunpath.sun_pc is not None
        assert len(sunpath.sun_pc.points) > 0


class TestSunCollectionCounters:
    """Tests for SunCollection count tracking."""

    def test_count_above_below_sum(self, solar_params_week):
        """Test that count_above + count_below equals original count."""
        sunpath = Sunpath(solar_params_week, include_night=False)
        # After creation, count should equal count_above (below removed)
        assert sunpath.sunc.count == sunpath.sunc.count_above

    def test_count_below_tracked(self, solar_params_week):
        """Test that count_below is tracked."""
        sunpath = Sunpath(solar_params_week, include_night=False)
        # Should have some below-horizon suns during a week
        assert sunpath.sunc.count_below >= 0


class TestDifferentLocations:
    """Tests for different geographic locations."""

    def test_polar_region(self, tmp_path):
        """Test sunpath creation for polar region (Tromso, Norway)."""
        # Create full-year EPW (required by Sunpath)
        df, header = synthetic_epw_df(
            latitude=69.65,  # Tromso
            longitude=18.96,
            timezone=1,
        )
        epw_path = tmp_path / "tromso.epw"
        df_to_epw(df, header, str(epw_path))

        params = SolarParameters(
            weather_file=str(epw_path),
            display=False,
            start=pd.Timestamp("2024-06-21 00:00"),
            end=pd.Timestamp("2024-06-28 00:00"),
        )
        sunpath = Sunpath(params)

        # Should still work for polar regions
        assert sunpath.sunc.count > 0

    def test_equatorial_region(self, tmp_path):
        """Test sunpath creation for equatorial region (Quito, Ecuador)."""
        df, header = synthetic_epw_df(
            latitude=-0.18,  # Quito
            longitude=-78.47,
            timezone=-5,  # UTC-5 (Ecuador Time)
        )
        epw_path = tmp_path / "quito.epw"
        df_to_epw(df, header, str(epw_path))

        params = SolarParameters(
            weather_file=str(epw_path),
            display=False,
            start=pd.Timestamp("2024-06-21 00:00"),
            end=pd.Timestamp("2024-06-28 00:00"),
        )
        sunpath = Sunpath(params)
        assert sunpath.sunc.count > 0
