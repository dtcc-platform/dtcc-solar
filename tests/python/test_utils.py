"""
Unit tests for dtcc_solar.utils module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from dtcc_core.model import Mesh
from dtcc_solar.utils import (
    Vec3,
    SolarParameters,
    SkyResults,
    SunResults,
    SunCollection,
    OutputCollection,
    SkyType,
    AnalysisType,
    SunMapping,
    Rays,
    hours_count,
    calc_face_normals,
    calc_face_areas,
    calc_face_mid_points,
    unitize,
    calc_vector_length,
    distance,
    concatenate_meshes,
    split_mesh_by_face_mask,
    split_mesh_by_vertical_faces,
    find_dup_faces,
    find_dup_vertices,
    is_mesh_valid,
)


class TestVec3:
    """Tests for Vec3 dataclass."""

    def test_vec3_creation(self):
        """Test Vec3 can be created with x, y, z coordinates."""
        v = Vec3(x=1.0, y=2.0, z=3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vec3_negative_values(self):
        """Test Vec3 with negative values."""
        v = Vec3(x=-1.5, y=-2.5, z=-3.5)
        assert v.x == -1.5
        assert v.y == -2.5
        assert v.z == -3.5

    def test_vec3_zero(self):
        """Test Vec3 with zero values."""
        v = Vec3(x=0.0, y=0.0, z=0.0)
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0


class TestEnums:
    """Tests for enum classes."""

    def test_sky_type_values(self):
        """Test SkyType enum values."""
        assert SkyType.TREGENZA_145 == 1
        assert SkyType.REINHART_578 == 2
        assert SkyType.REINHART_2305 == 3

    def test_analysis_type_values(self):
        """Test AnalysisType enum values."""
        assert AnalysisType.TWO_PHASE == 1
        assert AnalysisType.THREE_PHASE == 2

    def test_sun_mapping_values(self):
        """Test SunMapping enum values."""
        assert SunMapping.NONE == 1
        assert SunMapping.RADIANCE == 2
        assert SunMapping.SMOOTH_SMEAR == 3

    def test_rays_values(self):
        """Test Rays enum values."""
        assert Rays.BUNDLE_1 == 1
        assert Rays.BUNDLE_8 == 2


class TestSolarParameters:
    """Tests for SolarParameters dataclass."""

    def test_default_values(self, tmp_path):
        """Test SolarParameters default values."""
        epw_path = tmp_path / "test.epw"
        epw_path.touch()
        p = SolarParameters(weather_file=str(epw_path))
        assert p.display is True
        assert p.analysis_type == AnalysisType.TWO_PHASE
        assert p.sun_mapping == SunMapping.NONE

    def test_custom_values(self, tmp_path):
        """Test SolarParameters with custom values."""
        epw_path = tmp_path / "test.epw"
        epw_path.touch()
        p = SolarParameters(
            weather_file=str(epw_path),
            display=False,
            analysis_type=AnalysisType.THREE_PHASE,
            sun_mapping=SunMapping.RADIANCE,
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2024-12-31"),
        )
        assert p.display is False
        assert p.analysis_type == AnalysisType.THREE_PHASE
        assert p.sun_mapping == SunMapping.RADIANCE


class TestSunCollection:
    """Tests for SunCollection dataclass."""

    def test_empty_sun_collection(self):
        """Test empty SunCollection creation."""
        sc = SunCollection()
        assert sc.count == 0
        assert sc.count_above == 0
        assert sc.count_below == 0
        assert len(sc.time_stamps) == 0

    def test_sun_collection_with_data(self):
        """Test SunCollection with data."""
        sc = SunCollection()
        sc.count = 10
        sc.positions = np.random.rand(10, 3)
        sc.sun_vecs = np.random.rand(10, 3)
        sc.dni = np.random.rand(10)
        sc.dhi = np.random.rand(10)
        sc.zeniths = np.random.rand(10)
        assert sc.count == 10
        assert sc.positions.shape == (10, 3)


class TestSkyResults:
    """Tests for SkyResults dataclass."""

    def test_empty_sky_results(self):
        """Test empty SkyResults creation."""
        sr = SkyResults()
        assert sr.count == 0
        assert sr.matrix.shape == (0,)

    def test_sky_results_with_matrix(self):
        """Test SkyResults with matrix data."""
        sr = SkyResults()
        sr.count = 100
        sr.matrix = np.random.rand(145, 100)
        assert sr.count == 100
        assert sr.matrix.shape == (145, 100)


class TestSunResults:
    """Tests for SunResults dataclass."""

    def test_empty_sun_results(self):
        """Test empty SunResults creation."""
        sr = SunResults()
        assert sr.count == 0
        assert sr.matrix.shape == (0,)


class TestOutputCollection:
    """Tests for OutputCollection dataclass."""

    def test_empty_output_collection(self):
        """Test empty OutputCollection creation."""
        oc = OutputCollection()
        assert oc.shading_mesh is None
        assert oc.data_mask.shape == (0,)


class TestHoursCount:
    """Tests for hours_count function."""

    def test_hours_count_one_day(self):
        """Test hours_count for one day."""
        start = pd.Timestamp("2024-01-01 00:00")
        end = pd.Timestamp("2024-01-02 00:00")
        assert hours_count(start, end) == 24

    def test_hours_count_one_week(self):
        """Test hours_count for one week."""
        start = pd.Timestamp("2024-01-01 00:00")
        end = pd.Timestamp("2024-01-08 00:00")
        assert hours_count(start, end) == 168

    def test_hours_count_one_year(self):
        """Test hours_count for one year (non-leap year)."""
        start = pd.Timestamp("2023-01-01 00:00")
        end = pd.Timestamp("2024-01-01 00:00")
        assert hours_count(start, end) == 8760

    def test_hours_count_leap_year(self):
        """Test hours_count for one year (leap year)."""
        start = pd.Timestamp("2024-01-01 00:00")
        end = pd.Timestamp("2025-01-01 00:00")
        assert hours_count(start, end) == 8784

    def test_hours_count_zero(self):
        """Test hours_count for same start and end."""
        ts = pd.Timestamp("2024-01-01 00:00")
        assert hours_count(ts, ts) == 0

    def test_hours_count_invalid_order(self):
        """Test hours_count raises error when start > end."""
        start = pd.Timestamp("2024-01-02 00:00")
        end = pd.Timestamp("2024-01-01 00:00")
        with pytest.raises(ValueError, match="Start time must be before end time"):
            hours_count(start, end)


class TestVectorOperations:
    """Tests for vector utility functions."""

    def test_unitize_x_axis(self):
        """Test unitize with x-axis vector."""
        vec = np.array([5.0, 0.0, 0.0])
        result = unitize(vec)
        assert np.allclose(result, [1.0, 0.0, 0.0])

    def test_unitize_diagonal(self):
        """Test unitize with diagonal vector."""
        vec = np.array([1.0, 1.0, 1.0])
        result = unitize(vec)
        expected_length = 1.0 / np.sqrt(3.0)
        assert np.allclose(result, [expected_length, expected_length, expected_length])
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_calc_vector_length(self):
        """Test calc_vector_length."""
        vec = np.array([3.0, 4.0, 0.0])
        assert np.isclose(calc_vector_length(vec), 5.0)

    def test_calc_vector_length_3d(self):
        """Test calc_vector_length in 3D."""
        vec = np.array([1.0, 2.0, 2.0])
        assert np.isclose(calc_vector_length(vec), 3.0)

    def test_distance(self):
        """Test distance function."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([3.0, 4.0, 0.0])
        assert np.isclose(distance(v1, v2), 5.0)


class TestCalcFaceNormals:
    """Tests for calc_face_normals function."""

    def test_horizontal_triangle_normals(self, single_triangle_mesh):
        """Test face normals for horizontal triangle."""
        normals = calc_face_normals(single_triangle_mesh)
        assert normals.shape == (1, 3)
        # Horizontal triangle should have normal pointing up (0, 0, 1) or down
        assert np.isclose(abs(normals[0, 2]), 1.0, atol=1e-6)

    def test_normals_are_unit_vectors(self, small_cube_mesh):
        """Test that face normals are unit vectors."""
        normals = calc_face_normals(small_cube_mesh)
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_vertical_triangle_normal(self, vertical_triangle_mesh):
        """Test face normal for vertical triangle."""
        normals = calc_face_normals(vertical_triangle_mesh)
        # Normal should be perpendicular to the vertical face
        assert np.isclose(normals[0, 2], 0.0, atol=1e-6)


class TestCalcFaceAreas:
    """Tests for calc_face_areas function."""

    def test_unit_triangle_area(self, single_triangle_mesh):
        """Test area of unit right triangle."""
        areas = calc_face_areas(single_triangle_mesh)
        # Area of triangle with vertices (0,0,0), (1,0,0), (0,1,0) = 0.5
        assert np.isclose(areas[0], 0.5)

    def test_unit_square_area(self, unit_square_mesh):
        """Test area of unit square (two triangles)."""
        areas = calc_face_areas(unit_square_mesh)
        # Total area should be 1.0
        assert np.isclose(np.sum(areas), 1.0)

    def test_cube_face_areas(self, small_cube_mesh):
        """Test face areas of cube mesh."""
        areas = calc_face_areas(small_cube_mesh)
        # Each triangle face of unit cube should have area 0.5
        assert np.allclose(areas, 0.5)


class TestCalcFaceMidPoints:
    """Tests for calc_face_mid_points function."""

    def test_single_triangle_midpoint(self, single_triangle_mesh):
        """Test midpoint of single triangle."""
        midpoints = calc_face_mid_points(single_triangle_mesh)
        # Centroid of triangle (0,0,0), (1,0,0), (0,1,0)
        expected = np.array([[1 / 3, 1 / 3, 0]])
        assert np.allclose(midpoints, expected)

    def test_midpoints_count(self, small_cube_mesh):
        """Test number of midpoints equals number of faces."""
        midpoints = calc_face_mid_points(small_cube_mesh)
        assert len(midpoints) == len(small_cube_mesh.faces)


class TestConcatenateMeshes:
    """Tests for concatenate_meshes function."""

    def test_concatenate_two_triangles(self, single_triangle_mesh):
        """Test concatenating two identical triangles."""
        # Create a second triangle offset by 2 in x direction
        vertices2 = single_triangle_mesh.vertices + np.array([2, 0, 0])
        mesh2 = Mesh(vertices=vertices2, faces=single_triangle_mesh.faces.copy())

        combined = concatenate_meshes([single_triangle_mesh, mesh2])

        assert len(combined.vertices) == 6  # 3 + 3
        assert len(combined.faces) == 2  # 1 + 1

    def test_concatenate_single_mesh(self, single_triangle_mesh):
        """Test concatenating a single mesh."""
        combined = concatenate_meshes([single_triangle_mesh])
        assert len(combined.vertices) == len(single_triangle_mesh.vertices)
        assert len(combined.faces) == len(single_triangle_mesh.faces)


class TestSplitMeshByFaceMask:
    """Tests for split_mesh_by_face_mask function."""

    def test_split_unit_square(self, unit_square_mesh):
        """Test splitting unit square by face mask."""
        mask = np.array([True, False])
        mesh_in, mesh_out = split_mesh_by_face_mask(unit_square_mesh, mask)

        assert len(mesh_in.faces) == 1
        assert len(mesh_out.faces) == 1

    def test_split_all_true(self, unit_square_mesh):
        """Test split with all True mask."""
        mask = np.array([True, True])
        mesh_in, mesh_out = split_mesh_by_face_mask(unit_square_mesh, mask)

        assert len(mesh_in.faces) == 2
        assert len(mesh_out.faces) == 0

    def test_split_all_false(self, unit_square_mesh):
        """Test split with all False mask."""
        mask = np.array([False, False])
        mesh_in, mesh_out = split_mesh_by_face_mask(unit_square_mesh, mask)

        assert len(mesh_in.faces) == 0
        assert len(mesh_out.faces) == 2

    def test_split_invalid_mask_length(self, unit_square_mesh):
        """Test split with invalid mask length returns None."""
        mask = np.array([True])  # Wrong length
        result = split_mesh_by_face_mask(unit_square_mesh, mask)
        assert result == (None, None)


class TestSplitMeshByVerticalFaces:
    """Tests for split_mesh_by_vertical_faces function."""

    def test_split_cube_vertical(self, small_cube_mesh):
        """Test splitting cube by vertical faces."""
        horizontal, vertical, mask = split_mesh_by_vertical_faces(small_cube_mesh)

        # Cube has 2 horizontal faces (top, bottom) and 4 vertical faces
        # Each face is 2 triangles, so 4 horizontal triangles, 8 vertical
        assert len(horizontal.faces) + len(vertical.faces) == len(small_cube_mesh.faces)


class TestFindDuplicates:
    """Tests for find_dup_faces and find_dup_vertices functions."""

    def test_no_duplicate_faces(self, small_cube_mesh):
        """Test no duplicate faces in cube mesh."""
        dups = find_dup_faces(small_cube_mesh)
        assert len(dups) == 0

    def test_no_duplicate_vertices(self, small_cube_mesh):
        """Test no duplicate vertices in cube mesh."""
        dups = find_dup_vertices(small_cube_mesh)
        assert len(dups) == 0

    def test_with_duplicate_face(self):
        """Test detection of duplicate face."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 2]], dtype=int)  # Duplicate
        mesh = Mesh(vertices=vertices, faces=faces)

        dups = find_dup_faces(mesh)
        assert len(dups) == 1

    def test_with_duplicate_vertex(self):
        """Test detection of duplicate vertex."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]  # Last is duplicate
        ], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=int)
        mesh = Mesh(vertices=vertices, faces=faces)

        dups = find_dup_vertices(mesh)
        assert len(dups) >= 1


class TestIsMeshValid:
    """Tests for is_mesh_valid function."""

    def test_valid_mesh(self, small_cube_mesh):
        """Test that a normal cube mesh is valid."""
        assert is_mesh_valid(small_cube_mesh) is True

    def test_none_mesh(self):
        """Test that None mesh is invalid."""
        assert is_mesh_valid(None) is False

    def test_mesh_with_tiny_faces(self):
        """Test mesh with very small faces is invalid."""
        # Create a degenerate triangle with nearly zero area
        vertices = np.array([
            [0, 0, 0], [0.001, 0, 0], [0.0005, 0.001, 0]
        ], dtype=float)
        faces = np.array([[0, 1, 2]], dtype=int)
        mesh = Mesh(vertices=vertices, faces=faces)

        assert is_mesh_valid(mesh) is False
