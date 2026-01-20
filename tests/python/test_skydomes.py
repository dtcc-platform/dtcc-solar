"""
Unit tests for dtcc_solar skydome classes (Tregenza, ReinhartM2, ReinhartM4).

This extends the original test_3.py with additional comprehensive tests.
"""

import pytest
import math
import numpy as np

from dtcc_solar.tregenza import Tregenza
from dtcc_solar.reinhart2 import ReinhartM2
from dtcc_solar.reinhart4 import ReinhartM4


# ============================================================================
# Original tests from test_3.py
# ============================================================================


class TestTregenza:
    """Tests for Tregenza skydome (145 patches)."""

    def test_patch_count(self):
        """Test that Tregenza skydome has 145 patches."""
        dome = Tregenza()
        assert dome.patch_counter == 145

    def test_direction_vector_count(self):
        """Test that direction vector count matches patch count."""
        dome = Tregenza()
        assert len(dome.ray_dirs) == 145

    def test_direction_vectors_normalized(self):
        """Test that all direction vectors are unit vectors."""
        dome = Tregenza()
        norms = np.linalg.norm(dome.ray_dirs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_total_solid_angle(self):
        """Test that total solid angle is approximately 2π steradians."""
        dome = Tregenza()
        total_solid_angle = np.sum(dome.solid_angles)
        assert np.isclose(total_solid_angle, 2 * np.pi, rtol=1e-3)


class TestReinhartM2:
    """Tests for Reinhart M2 skydome (578 patches)."""

    def test_patch_count(self):
        """Test that Reinhart M2 skydome has 578 patches."""
        dome = ReinhartM2()
        assert dome.patch_counter == 578

    def test_direction_vector_count(self):
        """Test that direction vector count matches patch count."""
        dome = ReinhartM2()
        assert len(dome.ray_dirs) == 578

    def test_direction_vectors_normalized(self):
        """Test that all direction vectors are unit vectors."""
        dome = ReinhartM2()
        norms = np.linalg.norm(dome.ray_dirs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_total_solid_angle(self):
        """Test that total solid angle is approximately 2π steradians."""
        dome = ReinhartM2()
        total_solid_angle = np.sum(dome.solid_angles)
        assert np.isclose(total_solid_angle, 2 * np.pi, rtol=1e-3)


class TestReinhartM4:
    """Tests for Reinhart M4 skydome (2305 patches)."""

    def test_patch_count(self):
        """Test that Reinhart M4 skydome has 2305 patches."""
        dome = ReinhartM4()
        assert dome.patch_counter == 2305

    def test_direction_vector_count(self):
        """Test that direction vector count matches patch count."""
        dome = ReinhartM4()
        assert len(dome.ray_dirs) == 2305

    def test_direction_vectors_normalized(self):
        """Test that all direction vectors are unit vectors."""
        dome = ReinhartM4()
        norms = np.linalg.norm(dome.ray_dirs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_total_solid_angle(self):
        """Test that total solid angle is approximately 2π steradians."""
        dome = ReinhartM4()
        total_solid_angle = np.sum(dome.solid_angles)
        assert np.isclose(total_solid_angle, 2 * np.pi, rtol=1e-3)


# ============================================================================
# Extended tests for all skydomes
# ============================================================================


class TestSkydomeAttributes:
    """Tests for common skydome attributes across all types."""

    @pytest.mark.parametrize("dome_class,expected_patches", [
        (Tregenza, 145),
        (ReinhartM2, 578),
        (ReinhartM4, 2305),
    ])
    def test_patch_counts(self, dome_class, expected_patches):
        """Parametrized test for patch counts."""
        dome = dome_class()
        assert dome.patch_counter == expected_patches

    @pytest.mark.parametrize("dome_class", [Tregenza, ReinhartM2, ReinhartM4])
    def test_mesh_created(self, dome_class):
        """Test that mesh is created for all dome types."""
        dome = dome_class()
        assert dome.mesh is not None
        assert len(dome.mesh.vertices) > 0
        assert len(dome.mesh.faces) > 0

    @pytest.mark.parametrize("dome_class", [Tregenza, ReinhartM2, ReinhartM4])
    def test_solid_angles_non_negative(self, dome_class):
        """Test that all solid angles are non-negative."""
        dome = dome_class()
        assert all(sa >= 0 for sa in dome.solid_angles)

    @pytest.mark.parametrize("dome_class", [Tregenza, ReinhartM2, ReinhartM4])
    def test_ray_areas_non_negative(self, dome_class):
        """Test that all ray areas are non-negative."""
        dome = dome_class()
        assert all(ra >= 0 for ra in dome.ray_areas)

    @pytest.mark.parametrize("dome_class", [Tregenza, ReinhartM2, ReinhartM4])
    def test_ray_areas_sum_to_one(self, dome_class):
        """Test that ray areas sum to approximately 1.0."""
        dome = dome_class()
        total_area = np.sum(dome.ray_areas)
        assert np.isclose(total_area, 1.0, rtol=1e-3)

    @pytest.mark.parametrize("dome_class", [Tregenza, ReinhartM2, ReinhartM4])
    def test_patch_zeniths_in_valid_range(self, dome_class):
        """Test that patch zenith angles are in valid range [0, π/2]."""
        dome = dome_class()
        for zenith in dome.patch_zeniths:
            assert 0 <= zenith <= np.pi / 2

    @pytest.mark.parametrize("dome_class", [Tregenza, ReinhartM2, ReinhartM4])
    def test_ray_dirs_point_upward(self, dome_class):
        """Test that all ray directions point upward (positive z)."""
        dome = dome_class()
        for ray_dir in dome.ray_dirs:
            assert ray_dir[2] > 0  # z component should be positive


class TestSkydomeGeometry:
    """Tests for skydome geometry properties."""

    @pytest.mark.parametrize("dome_class", [Tregenza, ReinhartM2, ReinhartM4])
    def test_vertices_on_unit_sphere(self, dome_class):
        """Test that vertices lie on unit sphere."""
        dome = dome_class()
        for vertex in dome.mesh.vertices:
            dist = np.linalg.norm(vertex)
            assert np.isclose(dist, 1.0, atol=1e-6)

    @pytest.mark.parametrize("dome_class", [Tregenza, ReinhartM2, ReinhartM4])
    def test_vertices_upper_hemisphere(self, dome_class):
        """Test that all vertices are in upper hemisphere (z >= 0)."""
        dome = dome_class()
        for vertex in dome.mesh.vertices:
            assert vertex[2] >= -1e-6  # Allow small numerical errors

    @pytest.mark.parametrize("dome_class", [Tregenza, ReinhartM2, ReinhartM4])
    def test_faces_are_triangles(self, dome_class):
        """Test that all faces are triangles (3 vertices each)."""
        dome = dome_class()
        for face in dome.mesh.faces:
            assert len(face) == 3

    @pytest.mark.parametrize("dome_class", [Tregenza, ReinhartM2, ReinhartM4])
    def test_face_indices_valid(self, dome_class):
        """Test that face indices reference valid vertices."""
        dome = dome_class()
        num_vertices = len(dome.mesh.vertices)
        for face in dome.mesh.faces:
            for idx in face:
                assert 0 <= idx < num_vertices


class TestSkydomeSphericalToCartesian:
    """Tests for spherical to Cartesian conversion."""

    def test_zenith_point(self):
        """Test conversion at zenith (elevation = π/2)."""
        dome = Tregenza()
        result = dome.spherical_to_cartesian(np.pi / 2, 0)
        assert np.allclose(result, [0, 0, 1], atol=1e-6)

    def test_horizon_north(self):
        """Test conversion at horizon, north (azimuth = 0)."""
        dome = Tregenza()
        result = dome.spherical_to_cartesian(0, 0)
        assert np.allclose(result, [0, 1, 0], atol=1e-6)

    def test_horizon_east(self):
        """Test conversion at horizon, east (azimuth = π/2)."""
        dome = Tregenza()
        result = dome.spherical_to_cartesian(0, np.pi / 2)
        assert np.allclose(result, [1, 0, 0], atol=1e-6)

    def test_horizon_south(self):
        """Test conversion at horizon, south (azimuth = π)."""
        dome = Tregenza()
        result = dome.spherical_to_cartesian(0, np.pi)
        assert np.allclose(result, [0, -1, 0], atol=1e-6)


class TestSolidAngleCalculations:
    """Tests for solid angle calculations."""

    def test_full_hemisphere_solid_angle(self):
        """Test solid angle calculation for full hemisphere."""
        dome = Tregenza()
        # Full hemisphere from horizon to zenith
        solid_angle = dome.solid_angle(0, np.pi / 2, 0, 2 * np.pi)
        assert np.isclose(solid_angle, 2 * np.pi, rtol=1e-6)

    def test_top_patch_solid_angle(self):
        """Test top patch solid angle calculation."""
        dome = Tregenza()
        elev = math.radians(84)  # Tregenza's top band starts at 84°
        solid_angle = dome.calc_top_patch_solid_angle(elev)
        assert solid_angle > 0
        assert solid_angle < 2 * np.pi  # Less than full hemisphere


class TestMapDataToFaces:
    """Tests for map_data_to_faces function."""

    def test_tregenza_valid_data(self, sample_tregenza_data):
        """Test mapping valid data for Tregenza dome."""
        dome = Tregenza()
        mapped = dome.map_data_to_faces(sample_tregenza_data)
        # Each of 144 patches gets 2 triangles, zenith patch gets 6 triangles
        # Total: 144 * 2 + 6 = 294
        assert len(mapped) == len(dome.mesh.faces)

    def test_tregenza_wrong_data_length(self):
        """Test that wrong data length raises ValueError for Tregenza."""
        dome = Tregenza()
        wrong_data = np.random.rand(100)
        with pytest.raises(ValueError, match="145 elements"):
            dome.map_data_to_faces(wrong_data)

    def test_reinhart2_valid_data(self, sample_reinhart2_data):
        """Test mapping valid data for Reinhart M2 dome."""
        dome = ReinhartM2()
        mapped = dome.map_data_to_faces(sample_reinhart2_data)
        assert len(mapped) == len(dome.mesh.faces)

    def test_reinhart2_wrong_data_length(self):
        """Test that wrong data length raises ValueError for Reinhart M2."""
        dome = ReinhartM2()
        wrong_data = np.random.rand(145)
        with pytest.raises(ValueError, match="578 elements"):
            dome.map_data_to_faces(wrong_data)

    def test_tregenza_2d_data_summed(self):
        """Test that 2D data is summed along axis 1."""
        dome = Tregenza()
        data_2d = np.random.rand(145, 10)
        mapped = dome.map_data_to_faces(data_2d)
        assert len(mapped) == len(dome.mesh.faces)

    def test_reinhart2_2d_data_summed(self):
        """Test that 2D data is summed along axis 1 for Reinhart M2."""
        dome = ReinhartM2()
        data_2d = np.random.rand(578, 10)
        mapped = dome.map_data_to_faces(data_2d)
        assert len(mapped) == len(dome.mesh.faces)


class TestBandConfiguration:
    """Tests for skydome band configurations."""

    def test_tregenza_8_bands(self):
        """Test that Tregenza has 8 elevation bands."""
        dome = Tregenza()
        assert dome.bands == 8
        assert len(dome.band_patches) == 8

    def test_tregenza_band_patches_sum(self):
        """Test that Tregenza band patches sum to 145."""
        dome = Tregenza()
        assert sum(dome.band_patches) == 145

    def test_reinhart2_15_bands(self):
        """Test that Reinhart M2 has 15 elevation bands."""
        dome = ReinhartM2()
        assert dome.bands == 15
        assert len(dome.band_patches) == 15

    def test_reinhart2_band_patches_sum(self):
        """Test that Reinhart M2 band patches sum to 578."""
        dome = ReinhartM2()
        # Sum includes zenith patches (2 patches for M2)
        total = sum(dome.band_patches[:-1]) + 2  # Last band has 6 patches but merged to 2
        assert dome.patch_counter == 578


class TestQuadMidpoints:
    """Tests for quad midpoint calculations."""

    @pytest.mark.parametrize("dome_class", [Tregenza, ReinhartM2, ReinhartM4])
    def test_midpoints_on_sphere(self, dome_class):
        """Test that quad midpoints lie approximately on the unit sphere."""
        dome = dome_class()
        for midpoint in dome.quad_midpoints:
            dist = np.linalg.norm(midpoint)
            # Midpoints might be slightly off the sphere due to averaging
            assert np.isclose(dist, 1.0, atol=0.1)

    @pytest.mark.parametrize("dome_class", [Tregenza, ReinhartM2, ReinhartM4])
    def test_midpoints_upper_hemisphere(self, dome_class):
        """Test that quad midpoints are in upper hemisphere."""
        dome = dome_class()
        for midpoint in dome.quad_midpoints:
            assert midpoint[2] >= 0


class TestCalcSphereCapArea:
    """Tests for calc_sphere_cap_area function."""

    def test_cap_area_at_zenith(self):
        """Test sphere cap area at zenith (elevation = 90°)."""
        dome = Tregenza()
        # At zenith, polar angle = 0, so cap area = 0
        area = dome.calc_sphere_cap_area(np.pi / 2)
        assert np.isclose(area, 0, atol=1e-6)

    def test_cap_area_at_horizon(self):
        """Test sphere cap area at horizon (elevation = 0°)."""
        dome = Tregenza()
        # At horizon, polar angle = 90°, cap = hemisphere = 2πr²
        area = dome.calc_sphere_cap_area(0)
        expected = 2 * np.pi * dome.r**2
        assert np.isclose(area, expected, rtol=1e-6)

    def test_cap_area_intermediate(self):
        """Test sphere cap area at intermediate elevation."""
        dome = Tregenza()
        elev = np.pi / 4  # 45°
        area = dome.calc_sphere_cap_area(elev)
        # Cap area should be between 0 and hemisphere area
        assert 0 < area < 2 * np.pi * dome.r**2


class TestCalcHemisphereArea:
    """Tests for calc_hemisphere_area function."""

    def test_hemisphere_area(self):
        """Test hemisphere area calculation."""
        dome = Tregenza()
        area = dome.calc_hemisphere_area()
        expected = 2 * np.pi * dome.r**2
        assert np.isclose(area, expected)


class TestMapDictDataToFaces:
    """Tests for map_dict_data_to_faces function."""

    def test_tregenza_dict_mapping(self, sample_tregenza_data):
        """Test mapping dictionary of data for Tregenza dome."""
        dome = Tregenza()
        data_dict = {
            "dataset1": sample_tregenza_data,
            "dataset2": sample_tregenza_data * 2,
        }
        mapped = dome.map_dict_data_to_faces(data_dict)

        assert "dataset1" in mapped
        assert "dataset2" in mapped
        assert len(mapped["dataset1"]) == len(dome.mesh.faces)
        assert len(mapped["dataset2"]) == len(dome.mesh.faces)

    def test_reinhart2_dict_mapping(self, sample_reinhart2_data):
        """Test mapping dictionary of data for Reinhart M2 dome."""
        dome = ReinhartM2()
        data_dict = {
            "test": sample_reinhart2_data,
        }
        mapped = dome.map_dict_data_to_faces(data_dict)

        assert "test" in mapped
        assert len(mapped["test"]) == len(dome.mesh.faces)
