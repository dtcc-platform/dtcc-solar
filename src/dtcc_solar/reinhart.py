import math
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod
from dtcc_solar.skydome import Skydome
from dtcc_core.model import Mesh
from dtcc_solar.logging import info, debug, warning, error


class Reinhart(Skydome):

    def __init__(self):
        self.vertices = []
        self.faces = []
        self.ray_dirs = []
        self.patch_zeniths = []
        self.ray_areas = []
        self.solid_angles = []
        self.quad_midpoints = []

        self.patch_counter = 0
        self.bands = 15
        self.band_patches = [60, 60, 60, 60, 48, 48, 48, 48, 36, 36, 24, 24, 12, 12, 4]
        self.elevs_deg = [
            0.0,
            6.0,
            12.0,
            18.0,
            24.0,
            30.0,
            36.0,
            42.0,
            48.0,
            54.0,
            60.0,
            66.0,
            72.0,
            78.0,
            84.0,
        ]

    def create_mesh(self):

        dome_area = self.calc_hemisphere_area()
        elev_step = math.radians(6.0)

        for i in range(self.bands):
            patch_count = self.band_patches[i]
            elev1 = math.radians(self.elevs_deg[i])
            elev2 = elev1 + elev_step
            azim_step = 2.0 * math.pi / patch_count

            if elev1 != math.radians(84.0):
                for j in range(patch_count):
                    azim1 = j * azim_step
                    azim2 = (j + 1) * azim_step
                    mid_elev = (elev1 + elev2) / 2.0
                    mid_azim = (azim1 + azim2) / 2.0

                    self.create_mesh_quad(azim1, azim2, elev1, elev2)
                    ray_dir = self.spherical_to_cartesian(mid_elev, mid_azim)
                    self.ray_dirs.append(ray_dir)

                    solid_angle = self.solid_angle(elev1, elev2, azim1, azim2)
                    self.solid_angles.append(solid_angle)

                    self.patch_zeniths.append((math.pi / 2.0) - mid_elev)

                    patch_area = self.calc_sphere_patch_area(elev1, elev2, azim1, azim2)
                    self.ray_areas.append(patch_area / dome_area)
                    self.patch_counter += 1
            else:
                self.create_zenith_patch(elev1)

        mesh_vertices = np.array(self.vertices)
        mesh_faces = np.array(self.faces)
        self.mesh = Mesh(vertices=mesh_vertices, faces=mesh_faces)

        # Should be ≈ 6.283 (2π)
        tot_solid_angle = np.sum(self.solid_angles)
        info("Total solid angle: " + str(tot_solid_angle) + ", expected: 6.283")
        info(f"Reinhart skydome mesh with {self.patch_counter} patches created.")

    def create_zenith_patch(self, elev1):
        """Creating 4 zenith patches to cover the zenith area."""
        v_count = len(self.vertices)
        cap_area = self.calc_sphere_cap_area(elev1)
        dome_area = self.calc_hemisphere_area()
        elev2 = math.pi / 2.0

        for i in range(4):
            azimuth_step = math.pi / 2.0
            azim1 = i * azimuth_step
            azim2 = (i + 1) * azimuth_step

            pt1 = self.spherical_to_cartesian(elev1, azim1)
            pt2 = self.spherical_to_cartesian(elev1, azim1 + math.pi / 6.0)
            pt3 = self.spherical_to_cartesian(elev1, azim1 + 2.0 * math.pi / 6.0)
            pt4 = self.spherical_to_cartesian(elev1, azim2)
            pt5 = np.array([0.0, 0.0, self.r])  # Zenith point

            idx1 = v_count + i * 5
            idx2 = v_count + i * 5 + 1
            idx3 = v_count + i * 5 + 2
            idx4 = v_count + i * 5 + 3
            idx5 = v_count + i * 5 + 4

            self.vertices.extend([pt1, pt2, pt3, pt4, pt5])
            self.faces.extend(
                [
                    [idx1, idx2, idx5],
                    [idx2, idx3, idx5],
                    [idx3, idx4, idx5],
                ]
            )

            mid_azim = (azim1 + azim2) / 2.0
            mid_elev = (elev1 + elev2) / 2.0
            ray_dir = self.spherical_to_cartesian(mid_elev, mid_azim)
            self.ray_dirs.append(ray_dir)
            self.ray_areas.append((cap_area / 4.0) / dome_area)

            self.patch_zeniths.append((math.pi / 2.0) - mid_elev)

            solid_angle = self.solid_angle(elev1, elev2, azim1, azim2)
            self.solid_angles.append(solid_angle)
            self.patch_counter += 1

    def map_data_to_faces(self, data: np.ndarray) -> np.ndarray:
        if len(data) != 580:
            raise ValueError("Data must have 580 elements for Reinhart mapping.")

        data = np.array(data)
        if len(data.shape) == 2:
            data = np.sum(data, axis=1)

        data1, data2 = np.split(data, [576])

        data1 = np.repeat(data1, 2)  # Repeat data twice for quads (2 triangels)
        data2 = np.repeat(data2, 3)  # Repeat data 3 times for the 4 zenith patches
        data = np.append(data1, data2)  # Add data for last 4 triangels

        return data

    def map_dict_data_to_faces(self, dict_data: Dict) -> np.ndarray:

        mapped_dict = {}

        for key, data in dict_data.items():
            mapped_data = self.map_data_to_faces(data)
            mapped_dict[key] = mapped_data

        return mapped_dict
