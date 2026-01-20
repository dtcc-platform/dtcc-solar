"""
Unit tests for dtcc_solar.solar_engine module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from dtcc_core.model import Mesh
from dtcc_solar.solar_engine import SolarEngine, _require_solar
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.tregenza import Tregenza
from dtcc_solar.reinhart2 import ReinhartM2
from dtcc_solar.utils import (
    SolarParameters,
    OutputCollection,
    AnalysisType,
    SunMapping,
    concatenate_meshes,
)


# Check if solar bindings are available
try:
    from dtcc_solar import py_solar
    SOLAR_BINDINGS_AVAILABLE = True
except ImportError:
    SOLAR_BINDINGS_AVAILABLE = False


class TestSolarEngineCreation:
    """Tests for SolarEngine initialization."""

    def test_create_with_analysis_mesh_only(self, single_triangle_mesh):
        """Test creating engine with only analysis mesh."""
        engine = SolarEngine(single_triangle_mesh)

        assert engine.analysis_mesh is not None
        assert engine.shading_mesh is None
        assert engine.mesh is not None

    def test_create_with_shading_mesh(self, single_triangle_mesh, unit_square_mesh):
        """Test creating engine with analysis and shading meshes."""
        engine = SolarEngine(single_triangle_mesh, shading_mesh=unit_square_mesh)

        assert engine.analysis_mesh is not None
        assert engine.shading_mesh is not None
        assert engine.mesh is not None

    def test_combined_mesh_has_all_faces(
        self, single_triangle_mesh, unit_square_mesh
    ):
        """Test that combined mesh includes all faces."""
        engine = SolarEngine(single_triangle_mesh, shading_mesh=unit_square_mesh)

        expected_faces = len(single_triangle_mesh.faces) + len(unit_square_mesh.faces)
        assert len(engine.mesh.faces) == expected_faces


class TestFaceMask:
    """Tests for face mask generation."""

    def test_face_mask_analysis_only(self, single_triangle_mesh):
        """Test face mask when no shading mesh."""
        engine = SolarEngine(single_triangle_mesh)

        # All faces should be marked as analysis faces
        assert len(engine.face_mask) == len(single_triangle_mesh.faces)
        assert np.all(engine.face_mask == True)

    def test_face_mask_with_shading(self, single_triangle_mesh, unit_square_mesh):
        """Test face mask with shading mesh."""
        engine = SolarEngine(single_triangle_mesh, shading_mesh=unit_square_mesh)

        # First faces (analysis) should be True, rest (shading) should be False
        num_analysis = len(single_triangle_mesh.faces)
        num_shading = len(unit_square_mesh.faces)

        assert len(engine.face_mask) == num_analysis + num_shading
        assert np.all(engine.face_mask[:num_analysis] == True)
        assert np.all(engine.face_mask[num_analysis:] == False)

    def test_face_mask_sum_matches_analysis_faces(
        self, single_triangle_mesh, unit_square_mesh
    ):
        """Test that face mask sum equals analysis face count."""
        engine = SolarEngine(single_triangle_mesh, shading_mesh=unit_square_mesh)

        num_analysis_faces = np.sum(engine.face_mask)
        assert num_analysis_faces == len(single_triangle_mesh.faces)


class TestBoundsCalculation:
    """Tests for bounding box calculations."""

    def test_bounds_calculated(self, single_triangle_mesh):
        """Test that bounds are calculated."""
        engine = SolarEngine(single_triangle_mesh)

        assert engine.bb is not None
        assert hasattr(engine, "xmin")
        assert hasattr(engine, "xmax")
        assert hasattr(engine, "ymin")
        assert hasattr(engine, "ymax")
        assert hasattr(engine, "zmin")
        assert hasattr(engine, "zmax")

    def test_bounds_values_correct(self, single_triangle_mesh):
        """Test that bound values match mesh extents."""
        engine = SolarEngine(single_triangle_mesh)

        # Single triangle: (0,0,0), (1,0,0), (0,1,0)
        assert engine.xmin == 0
        assert engine.xmax == 1
        assert engine.ymin == 0
        assert engine.ymax == 1
        assert engine.zmin == 0
        assert engine.zmax == 0


class TestSunpathRadius:
    """Tests for sunpath radius calculation."""

    def test_sunpath_radius_positive(self, small_cube_mesh):
        """Test that sunpath radius is positive."""
        engine = SolarEngine(small_cube_mesh)
        assert engine.sunpath_radius > 0

    def test_sunpath_radius_covers_mesh(self, small_cube_mesh):
        """Test that sunpath radius is large enough to cover mesh."""
        engine = SolarEngine(small_cube_mesh)

        # Radius should be based on mesh diagonal
        mesh_diagonal = np.sqrt(
            engine.bb.width**2 + engine.bb.height**2 +
            (engine.zmax - engine.zmin)**2
        )
        assert engine.sunpath_radius >= mesh_diagonal / 2


class TestCenterMesh:
    """Tests for mesh centering option."""

    def test_center_mesh_moves_to_origin(self, small_cube_mesh):
        """Test that center_mesh=True moves mesh center to origin."""
        engine = SolarEngine(small_cube_mesh, center_mesh=True)

        # After centering, mesh center should be at origin
        center_x = (engine.xmin + engine.xmax) / 2
        center_y = (engine.ymin + engine.ymax) / 2

        assert np.isclose(center_x, 0, atol=0.1)
        assert np.isclose(center_y, 0, atol=0.1)


class TestSolarBindingsRequired:
    """Tests for solar bindings requirement."""

    def test_require_solar_raises_when_missing(self):
        """Test that _require_solar raises error when bindings missing."""
        # This test only makes sense when bindings are missing
        if SOLAR_BINDINGS_AVAILABLE:
            # Just verify function works when bindings present
            result = _require_solar()
            assert result is not None
        else:
            with pytest.raises(RuntimeError, match="Solar ray-tracing bindings"):
                _require_solar()


@pytest.mark.skipif(
    not SOLAR_BINDINGS_AVAILABLE,
    reason="C++ solar bindings not available"
)
class TestSolarEngineAnalysis:
    """Tests for solar analysis execution (requires C++ bindings)."""

    def test_2_phase_analysis_returns_output_collection(
        self, single_triangle_mesh, solar_params_week
    ):
        """Test that 2-phase analysis returns OutputCollection."""
        engine = SolarEngine(single_triangle_mesh)
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()

        output = engine.run_2_phase_analysis(sunpath, skydome, solar_params_week)

        assert isinstance(output, OutputCollection)

    def test_2_phase_analysis_has_results(
        self, single_triangle_mesh, solar_params_week
    ):
        """Test that 2-phase analysis populates result fields."""
        engine = SolarEngine(single_triangle_mesh)
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()

        output = engine.run_2_phase_analysis(sunpath, skydome, solar_params_week)

        assert output.total_irradiance is not None
        assert len(output.total_irradiance) > 0
        assert output.sky_view_factor is not None
        assert len(output.sky_view_factor) > 0

    def test_3_phase_analysis_returns_output_collection(
        self, single_triangle_mesh, synthetic_epw_path
    ):
        """Test that 3-phase analysis returns OutputCollection."""
        params = SolarParameters(
            weather_file=synthetic_epw_path,
            display=False,
            analysis_type=AnalysisType.THREE_PHASE,
            start=pd.Timestamp("2024-06-21 00:00"),
            end=pd.Timestamp("2024-06-28 00:00"),
        )

        engine = SolarEngine(single_triangle_mesh)
        sunpath = Sunpath(params)
        skydome = Tregenza()

        output = engine.run_3_phase_analysis(sunpath, skydome, params)

        assert isinstance(output, OutputCollection)

    def test_3_phase_analysis_has_sun_hours(
        self, single_triangle_mesh, synthetic_epw_path
    ):
        """Test that 3-phase analysis includes sun hours."""
        params = SolarParameters(
            weather_file=synthetic_epw_path,
            display=False,
            analysis_type=AnalysisType.THREE_PHASE,
            start=pd.Timestamp("2024-06-21 00:00"),
            end=pd.Timestamp("2024-06-28 00:00"),
        )

        engine = SolarEngine(single_triangle_mesh)
        sunpath = Sunpath(params)
        skydome = Tregenza()

        output = engine.run_3_phase_analysis(sunpath, skydome, params)

        assert output.sun_hours is not None
        assert len(output.sun_hours) > 0

    def test_irradiance_values_non_negative(
        self, single_triangle_mesh, solar_params_week
    ):
        """Test that irradiance values are non-negative."""
        engine = SolarEngine(single_triangle_mesh)
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()

        output = engine.run_2_phase_analysis(sunpath, skydome, solar_params_week)

        assert np.all(output.total_irradiance >= 0)

    def test_sky_view_factor_in_range(
        self, single_triangle_mesh, solar_params_week
    ):
        """Test that sky view factor is in valid range [0, 1]."""
        engine = SolarEngine(single_triangle_mesh)
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()

        output = engine.run_2_phase_analysis(sunpath, skydome, solar_params_week)

        assert np.all(output.sky_view_factor >= 0)
        assert np.all(output.sky_view_factor <= 1)

    def test_run_analysis_dispatches_correctly(
        self, single_triangle_mesh, solar_params_week
    ):
        """Test that run_analysis dispatches to correct method."""
        engine = SolarEngine(single_triangle_mesh)
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()

        # 2-phase (default)
        output = engine.run_analysis(sunpath, skydome, solar_params_week)
        assert isinstance(output, OutputCollection)


@pytest.mark.skipif(
    not SOLAR_BINDINGS_AVAILABLE,
    reason="C++ solar bindings not available"
)
class TestSolarEngineWithShading:
    """Tests for solar analysis with shading mesh."""

    def test_shading_reduces_irradiance(
        self, horizontal_upward_triangle, synthetic_epw_path
    ):
        """Test that adding shading mesh reduces irradiance."""
        params = SolarParameters(
            weather_file=synthetic_epw_path,
            display=False,
            analysis_type=AnalysisType.TWO_PHASE,
            start=pd.Timestamp("2024-06-21 00:00"),
            end=pd.Timestamp("2024-06-28 00:00"),
        )

        skydome = Tregenza()

        # Without shading
        engine_no_shade = SolarEngine(horizontal_upward_triangle)
        sunpath = Sunpath(params)
        output_no_shade = engine_no_shade.run_2_phase_analysis(
            sunpath, skydome, params
        )
        irr_no_shade = output_no_shade.total_irradiance[0]

        # With shading (large panel above the triangle)
        shading_vertices = np.array([
            [-5, -5, 5], [5, -5, 5], [5, 5, 5], [-5, 5, 5]
        ], dtype=float)
        shading_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        shading_mesh = Mesh(vertices=shading_vertices, faces=shading_faces)

        engine_with_shade = SolarEngine(
            horizontal_upward_triangle, shading_mesh=shading_mesh
        )
        sunpath2 = Sunpath(params)
        output_with_shade = engine_with_shade.run_2_phase_analysis(
            sunpath2, skydome, params
        )

        # Extract irradiance for analysis face only
        irr_with_shade = output_with_shade.total_irradiance[
            output_with_shade.data_mask
        ][0]

        # Shading should reduce irradiance (or at least not increase it)
        assert irr_with_shade <= irr_no_shade


@pytest.mark.skipif(
    not SOLAR_BINDINGS_AVAILABLE,
    reason="C++ solar bindings not available"
)
class TestSkyViewFactor:
    """Tests for sky view factor calculations."""

    def test_horizontal_unobstructed_svf_near_one(
        self, horizontal_upward_triangle, solar_params_week
    ):
        """Test that unobstructed horizontal surface has SVF close to 1.0."""
        engine = SolarEngine(horizontal_upward_triangle)
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()

        output = engine.run_2_phase_analysis(sunpath, skydome, solar_params_week)

        # Unobstructed upward-facing surface should see nearly full sky
        # Allow some tolerance for discretization and numerical errors
        svf = output.sky_view_factor[0]
        assert svf > 0.9


class TestOutputCollection:
    """Tests for OutputCollection structure."""

    @pytest.mark.skipif(
        not SOLAR_BINDINGS_AVAILABLE,
        reason="C++ solar bindings not available"
    )
    def test_output_has_mesh(self, single_triangle_mesh, solar_params_week):
        """Test that output contains mesh."""
        engine = SolarEngine(single_triangle_mesh)
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()

        output = engine.run_2_phase_analysis(sunpath, skydome, solar_params_week)

        assert output.mesh is not None

    @pytest.mark.skipif(
        not SOLAR_BINDINGS_AVAILABLE,
        reason="C++ solar bindings not available"
    )
    def test_output_has_data_mask(self, single_triangle_mesh, solar_params_week):
        """Test that output contains data mask."""
        engine = SolarEngine(single_triangle_mesh)
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()

        output = engine.run_2_phase_analysis(sunpath, skydome, solar_params_week)

        assert output.data_mask is not None
        assert len(output.data_mask) == len(output.mesh.faces)

    @pytest.mark.skipif(
        not SOLAR_BINDINGS_AVAILABLE,
        reason="C++ solar bindings not available"
    )
    def test_output_has_sky_results(self, single_triangle_mesh, solar_params_week):
        """Test that output contains sky results."""
        engine = SolarEngine(single_triangle_mesh)
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()

        output = engine.run_2_phase_analysis(sunpath, skydome, solar_params_week)

        assert output.sky_results is not None

    @pytest.mark.skipif(
        not SOLAR_BINDINGS_AVAILABLE,
        reason="C++ solar bindings not available"
    )
    def test_output_has_sun_results(self, single_triangle_mesh, solar_params_week):
        """Test that output contains sun results."""
        engine = SolarEngine(single_triangle_mesh)
        sunpath = Sunpath(solar_params_week)
        skydome = Tregenza()

        output = engine.run_2_phase_analysis(sunpath, skydome, solar_params_week)

        assert output.sun_results is not None


class TestMultipleMeshes:
    """Tests for analysis with multiple mesh configurations."""

    def test_multiple_analysis_faces(self, unit_square_mesh):
        """Test engine with mesh having multiple faces."""
        engine = SolarEngine(unit_square_mesh)

        assert len(engine.face_mask) == 2
        assert np.sum(engine.face_mask) == 2

    def test_cube_mesh(self, small_cube_mesh):
        """Test engine with cube mesh (12 triangles)."""
        engine = SolarEngine(small_cube_mesh)

        assert len(engine.face_mask) == 12
        assert np.sum(engine.face_mask) == 12
