import numpy as np
import math
from typing import List
from abc import ABC, abstractmethod
from dtcc_core.model import Mesh, MultiLineString
from dtcc_viewer import Scene, Window
from dtcc_core.model import Mesh, LineString, MultiLineString, PointCloud
from dtcc_solar.utils import create_ls_circle


class Skydome(ABC):

    r: float = 1.0  # Radius of the skydome, default is 1.0

    mesh: Mesh
    mls: MultiLineString
    sky_vector_matrix: np.ndarray
    per_rel_lumiance: np.ndarray
    absolute_luminance: np.ndarray

    faces: List[List[int]]
    vertices: List[List[float]]
    ray_dirs: List[List[float]]
    ray_areas: List[float]
    solid_angles: List[float]
    patch_zeniths: List[float]
    quad_midpoints: List[List[float]]

    bands: int
    patch_counter: int
    band_patches: List[int]
    elevs_deg: List[float]

    @abstractmethod
    def create_mesh(self):
        """Create a skydome mesh."""
        pass

    @abstractmethod
    def create_zenith_patch(self):
        """Create the patch where zenith is 0 for the skydome mesh."""
        pass

    @abstractmethod
    def map_data_to_faces(self, data: np.ndarray) -> np.ndarray:
        """Map data to the faces of the skydome mesh."""
        pass

    def create_mesh_quad(self, azim, next_azim, elev, next_elev) -> None:
        pt1 = self.spherical_to_cartesian(elev, azim)
        pt2 = self.spherical_to_cartesian(next_elev, azim)
        pt3 = self.spherical_to_cartesian(elev, next_azim)
        pt4 = self.spherical_to_cartesian(next_elev, next_azim)

        self.vertices.extend([pt1, pt2, pt3, pt4])

        v_count = len(self.vertices)

        idx0 = v_count - 4
        idx1 = v_count - 3
        idx2 = v_count - 2
        idx3 = v_count - 1

        self.faces.append([idx0, idx1, idx2])
        self.faces.append([idx1, idx3, idx2])

        mp_x = (pt1[0] + pt2[0] + pt3[0] + pt4[0]) / 4.0
        mp_y = (pt1[1] + pt2[1] + pt3[1] + pt4[1]) / 4.0
        mp_z = (pt1[2] + pt2[2] + pt3[2] + pt4[2]) / 4.0

        self.quad_midpoints.append([mp_x, mp_y, mp_z])

    def solid_angle(self, elev1, elev2, azim1, azim2) -> float:
        """
        Compute solid angle (steradians) of a spherical rectangle patch on a unit hemisphere,
        using elevation and azimuth bounds in **radians**.

        Parameters:
            elev1, elev2 : float
                Elevation bounds (in radians), from horizon (0) up to zenith (π/2)
            azim1, azim2 : float
                Azimuth bounds (in degrgees), range [0, 2π]

        Returns:
            float : solid angle in steradians
        """

        return (azim2 - azim1) * (np.sin(elev2) - np.sin(elev1))

    def calc_top_patch_solid_angle(self, elev) -> float:
        zenith = math.pi / 2.0 - elev
        solid_angle = 2 * np.pi * (1 - np.cos(zenith))
        return solid_angle

    def spherical_to_cartesian(self, elev, azim) -> List[float]:
        x = self.r * math.cos(elev) * math.cos(azim)
        y = self.r * math.cos(elev) * math.sin(azim)
        z = self.r * math.sin(elev)
        return [x, y, z]

    def calc_sphere_cap_area(self, elevation) -> float:
        polar_angle = math.pi / 2.0 - elevation
        return 2.0 * math.pi * (self.r**2) * (1 - math.cos(polar_angle))

    def create_dirvector_mls(self, line_s, line_dir) -> None:
        line_strings = []
        line_s = np.array(line_s)
        line_dir = np.array(line_dir)

        for start, direction in zip(line_s, line_dir):
            end = start + direction / np.linalg.norm(direction)
            vertices = []
            vertices.append(start)
            vertices.append(end)
            vertices = np.array(vertices)
            ls = LineString(vertices=vertices)
            line_strings.append(ls)

        self.mls = MultiLineString(linestrings=line_strings)

    def calc_hemisphere_area(self) -> float:
        return 2.0 * math.pi * (self.r**2)

    def calc_sphere_patch_area(self, elev1, elev2, azim1, azim2) -> float:
        return (self.r**2) * abs((azim2 - azim1) * (math.sin(elev2) - math.sin(elev1)))

    def view(self, name: str, data=None, sun_pos_pc: PointCloud = None) -> None:
        ls_cirlce = create_ls_circle(np.array([0, 0, 0]), self.r * 2, 100)
        window = Window(1200, 800)
        scene = Scene()
        scene.add_mesh(name, self.mesh, data=data)
        scene.add_pointcloud("Sun positions", sun_pos_pc, size=0.01)
        scene.add_linestring("Skydome circle", ls_cirlce)
        window.render(scene)
