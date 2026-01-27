"""
Shared pytest fixtures for dtcc-solar unit tests.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from dtcc_core.model import Mesh
from dtcc_solar.synthetic_data import synthetic_epw_df, df_to_epw
from dtcc_solar.utils import SolarParameters, AnalysisType, SunMapping
from dtcc_solar.tregenza import Tregenza
from dtcc_solar.reinhart2 import ReinhartM2
from dtcc_solar.reinhart4 import ReinhartM4


# ============================================================================
# Mesh Fixtures
# ============================================================================


@pytest.fixture
def single_triangle_mesh():
    """A single horizontal triangle at z=0."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    return Mesh(vertices=vertices, faces=faces)


@pytest.fixture
def unit_square_mesh():
    """A unit square made of two triangles at z=0."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return Mesh(vertices=vertices, faces=faces)


@pytest.fixture
def small_cube_mesh():
    """A small cube mesh (6 faces, 12 triangles)."""
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],  # bottom
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],  # top
        ],
        dtype=float,
    )
    faces = np.array(
        [
            # bottom
            [0, 2, 1],
            [0, 3, 2],
            # top
            [4, 5, 6],
            [4, 6, 7],
            # front
            [0, 1, 5],
            [0, 5, 4],
            # back
            [2, 3, 7],
            [2, 7, 6],
            # left
            [0, 4, 7],
            [0, 7, 3],
            # right
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=int,
    )
    return Mesh(vertices=vertices, faces=faces)


@pytest.fixture
def vertical_triangle_mesh():
    """A vertical triangle (facing East, perpendicular to ground)."""
    vertices = np.array([[0, 0, 0], [0, 1, 0], [0, 0.5, 1]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    return Mesh(vertices=vertices, faces=faces)


@pytest.fixture
def horizontal_upward_triangle():
    """A horizontal triangle with normal pointing upward (+Z)."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    return Mesh(vertices=vertices, faces=faces)


# ============================================================================
# Weather Data Fixtures
# ============================================================================


@pytest.fixture
def synthetic_epw_df_full_year():
    """Synthetic EPW DataFrame for full year (required by Sunpath)."""
    df, header = synthetic_epw_df(
        latitude=59.33,
        longitude=18.06,
        timezone=1,
    )
    return df, header


@pytest.fixture
def synthetic_epw_path(synthetic_epw_df_full_year, tmp_path):
    """Temporary path to a full-year synthetic EPW file.

    Note: Sunpath requires full-year EPW files starting from January 1st.
    """
    df, header = synthetic_epw_df_full_year
    epw_path = tmp_path / "synthetic_test.epw"
    df_to_epw(df, header, str(epw_path))
    return str(epw_path)


# ============================================================================
# Solar Parameter Fixtures
# ============================================================================


@pytest.fixture
def solar_params_day(synthetic_epw_path):
    """Solar parameters for a single day analysis."""
    return SolarParameters(
        weather_file=synthetic_epw_path,
        display=False,
        analysis_type=AnalysisType.TWO_PHASE_1D,
        sun_mapping=SunMapping.NONE,
        start=pd.Timestamp("2024-06-21 00:00"),
        end=pd.Timestamp("2024-06-22 00:00"),
    )


@pytest.fixture
def solar_params_week(synthetic_epw_path):
    """Solar parameters for a week-long analysis."""
    return SolarParameters(
        weather_file=synthetic_epw_path,
        display=False,
        analysis_type=AnalysisType.TWO_PHASE_1D,
        sun_mapping=SunMapping.NONE,
        start=pd.Timestamp("2024-06-21 00:00"),
        end=pd.Timestamp("2024-06-28 00:00"),
    )


# ============================================================================
# Skydome Fixtures
# ============================================================================


@pytest.fixture
def tregenza_dome():
    """Tregenza skydome (145 patches)."""
    return Tregenza()


@pytest.fixture
def reinhart2_dome():
    """Reinhart M2 skydome (578 patches)."""
    return ReinhartM2()


@pytest.fixture
def reinhart4_dome():
    """Reinhart M4 skydome (2305 patches)."""
    return ReinhartM4()


@pytest.fixture(params=["tregenza", "reinhart2", "reinhart4"])
def any_skydome(request):
    """Parametrized fixture to test all skydome types."""
    if request.param == "tregenza":
        return Tregenza()
    elif request.param == "reinhart2":
        return ReinhartM2()
    elif request.param == "reinhart4":
        return ReinhartM4()


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_tregenza_data():
    """Sample data array with 145 elements (matching Tregenza patches)."""
    return np.random.rand(145)


@pytest.fixture
def sample_reinhart2_data():
    """Sample data array with 578 elements (matching Reinhart M2 patches)."""
    return np.random.rand(578)


@pytest.fixture
def sample_reinhart4_data():
    """Sample data array with 2305 elements (matching Reinhart M4 patches)."""
    return np.random.rand(2305)


# ============================================================================
# Helper Functions
# ============================================================================


@pytest.fixture
def solar_bindings_available():
    """Check if C++ solar bindings are available."""
    try:
        from dtcc_solar import py_solar

        return True
    except ImportError:
        return False
