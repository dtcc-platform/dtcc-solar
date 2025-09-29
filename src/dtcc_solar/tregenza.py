import math
import numpy as np
from typing import Dict
from dtcc_solar.skydome import Skydome
from dtcc_core.model import Mesh
from dtcc_solar.logging import info, debug, warning, error


class Tregenza(Skydome):

    def __init__(self):
        self.vertices = []
        self.faces = []
        self.ray_dirs = []
        self.patch_zeniths = []
        self.ray_areas = []
        self.solid_angles = []
        self.quad_midpoints = []
        self.bands = 8
        self.patch_counter = 0
        self.band_patches = [30, 30, 24, 24, 18, 12, 6, 1]
        self.elevs_deg = [0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0]

        self.create_mesh()

    def create_mesh(self):

        elev_step = math.radians(12.0)
        dome_area = self.calc_hemisphere_area()

        for i in range(self.bands):
            patch_count = self.band_patches[i]
            elev1 = math.radians(self.elevs_deg[i])
            elev2 = elev1 + elev_step
            azim_step = 2 * math.pi / patch_count

            if elev1 != math.radians(84.0):
                for j in range(patch_count):
                    step = 2 * math.pi / patch_count
                    azim1 = j * step - 0.5 * step
                    azim2 = (j + 1) * step - 0.5 * step
                    mid_azim = j * step
                    mid_elev = (elev1 + elev2) / 2.0

                    self.create_mesh_quad(azim1, azim2, elev1, elev2)

                    ray_dir = self.spherical_to_cartesian(mid_elev, mid_azim)
                    self.ray_dirs.append(ray_dir)

                    self.patch_zeniths.append((math.pi / 2.0) - mid_elev)

                    solid_angle = self.solid_angle(elev1, elev2, azim1, azim2)
                    self.solid_angles.append(solid_angle)

                    patch_area = self.calc_sphere_patch_area(elev1, elev2, azim1, azim2)
                    self.ray_areas.append(patch_area / dome_area)
                    self.patch_counter += 1
            else:
                self.create_zenith_patch(elev1)

        mesh_vertices = np.array(self.vertices)
        mesh_faces = np.array(self.faces)
        self.mesh = Mesh(vertices=mesh_vertices, faces=mesh_faces)

        # Should be ≈ 6.283 (2π)
        tot_solid_angle = np.round(np.sum(self.solid_angles), 3)
        info("-----------------------------------------------------")
        info(f"Tregenza skydome created:")
        info(f"  Number of patches: {self.patch_counter}")
        info(f"  Number of direction vectors: {len(self.ray_dirs)}")
        info(f"  Total solid angle: ~{tot_solid_angle}, expected: ~6.283")
        info("-----------------------------------------------------")

    def create_zenith_patch(self, elev):
        """Creating one zenith patch as a cone with 6 triangular faces"""
        v_count = len(self.vertices)
        dome_area = self.calc_hemisphere_area()

        ray_dir = [0.0, 0.0, 1.0]
        self.ray_dirs.append(ray_dir)

        zenith_angle = np.pi / 2.0 - elev
        self.patch_zeniths.append(zenith_angle)

        solid_angle = self.calc_top_patch_solid_angle(elev)
        self.solid_angles.append(solid_angle)

        cap_area = self.calc_sphere_cap_area(elev)
        self.ray_areas.append(cap_area / dome_area)

        for i in range(6):
            azimuth_step = 2 * math.pi / 6
            azim = i * azimuth_step
            pt = self.spherical_to_cartesian(elev, azim)
            self.vertices.append(pt)

            idx1 = v_count + i
            idx2 = v_count + (i + 1) % 6
            idx3 = v_count + 6  # Zenith point

            self.faces.append([idx1, idx2, idx3])

        self.vertices.append([0.0, 0.0, 1.0])  # Zenith point
        self.patch_counter += 1

    def map_data_to_faces(self, data: np.ndarray) -> np.ndarray:
        if len(data) != 145:
            raise ValueError("Data must have 145 elements for Tregenza mapping.")

        data = np.array(data)
        if len(data.shape) == 2:
            data = np.sum(data, axis=1)

        last = data[-1]
        data = np.repeat(data, 2)
        data = np.append(data, [last, last, last, last])

        return data

    def map_dict_data_to_faces(self, dict_data: Dict) -> np.ndarray:

        mapped_dict = {}

        for key, data in dict_data.items():
            mapped_data = self.map_data_to_faces(data)
            mapped_dict[key] = mapped_data

        return mapped_dict
