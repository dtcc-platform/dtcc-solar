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
        self.band_patches = [60, 60, 60, 60, 48, 48, 48, 48, 36, 36, 24, 24, 12, 12, 6]
        # Note: last band now has 2 patches resulting in 578 patches after zenith merge
        self.elevs_deg = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]

        self.create_mesh()

    def create_mesh(self):
        dome_area = self.calc_hemisphere_area()
        elev_step = math.radians(6.0)

        for i in range(self.bands):
            patch_count = self.band_patches[i]
            elev1 = math.radians(self.elevs_deg[i])
            elev2 = elev1 + elev_step
            azim_step = 2.0 * math.pi / patch_count

            if elev1 < math.radians(84.0):  # regular bands
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
                # Zenith band = 2 patches
                self.create_zenith_patch(elev1, elev2)

        mesh_vertices = np.array(self.vertices)
        mesh_faces = np.array(self.faces)
        self.mesh = Mesh(vertices=mesh_vertices, faces=mesh_faces)

        tot_solid_angle = np.round(np.sum(self.solid_angles), 3)
        info("-----------------------------------------------------")
        info(f"Reinhart skydome created (Radiance 578 style):")
        info(f"  Number of patches: {self.patch_counter}")
        info(f"  Number of direction vectors: {len(self.ray_dirs)}")
        info(f"  Total solid angle: ~{tot_solid_angle}, expected: ~6.283")
        info("-----------------------------------------------------")

    def create_zenith_patch(self, elev1, elev2):
        """
        Create 2 zenith patches for the band 84–90°,
        consistent with Radiance's Reinhart (578 patches).
        """
        dome_area = self.calc_hemisphere_area()
        cap_area = self.calc_sphere_cap_area(elev1)  # spherical cap above elev1
        azim_step = math.pi  # 180° per patch → 2 patches
        v_count = len(self.vertices)

        for i in range(2):
            azim1 = i * azim_step
            azim2 = (i + 1) * azim_step
            mid_elev = (elev1 + elev2) / 2.0
            mid_azim = (azim1 + azim2) / 2.0

            # Create quad for this zenith patch
            self.create_zenith_mesh(i, v_count, elev1, elev2, azim1, azim2)

            # Direction vector at patch midpoint
            ray_dir = self.spherical_to_cartesian(mid_elev, mid_azim)
            self.ray_dirs.append(ray_dir)

            # Solid angle of patch
            self.solid_angles.append(self.solid_angle(elev1, elev2, azim1, azim2))

            # Zenith angle of midpoint
            self.patch_zeniths.append((math.pi / 2.0) - mid_elev)

            # Relative area (cap area equally split into 2 patches)
            self.ray_areas.append((cap_area / 2.0) / dome_area)

            self.patch_counter += 1

    def map_data_to_faces(self, data: np.ndarray) -> np.ndarray:
        """
        Map 578 patch values to per-face values for visualization.
        Each quadrilateral patch was triangulated into 2 faces,
        so we repeat each patch value twice.
        """
        if len(data) != 578:
            raise ValueError("Data must have 578 elements for Reinhart mapping.")

        data = np.array(data)
        if len(data.shape) == 2:
            # Collapse time axis if given (sum over columns)
            data = np.sum(data, axis=1)

        # Each patch corresponds to 2 triangles
        last = data[-1]
        scnd_last = data[-2]

        # the 576 first patches have 2 triangles for representation
        data = np.repeat(data[0:576], 2)

        # the last and second to last have 3 triangles for representation
        data = np.append(data, [scnd_last, scnd_last, scnd_last, last, last, last])

        return data

    def map_dict_data_to_faces(self, dict_data: Dict) -> np.ndarray:
        """
        Apply map_data_to_faces for a dictionary of datasets.
        """
        mapped_dict = {}
        for key, data in dict_data.items():
            mapped_dict[key] = self.map_data_to_faces(data)
        return mapped_dict

    def create_zenith_mesh(self, index, v_count, elev1, elev2, azim1, azim2):

        # Creating 3 triangles per half zenith patch

        pt1 = self.spherical_to_cartesian(elev1, azim1)
        pt2 = self.spherical_to_cartesian(elev1, azim1 + math.pi / 3.0)
        pt3 = self.spherical_to_cartesian(elev1, azim1 + 2.0 * math.pi / 3.0)
        pt4 = self.spherical_to_cartesian(elev1, azim2)
        pt5 = np.array([0.0, 0.0, self.r])  # Zenith point

        idx1 = v_count + index * 5
        idx2 = v_count + index * 5 + 1
        idx3 = v_count + index * 5 + 2
        idx4 = v_count + index * 5 + 3
        idx5 = v_count + index * 5 + 4

        self.vertices.extend([pt1, pt2, pt3, pt4, pt5])
        self.faces.extend(
            [
                [idx1, idx2, idx5],
                [idx2, idx3, idx5],
                [idx3, idx4, idx5],
            ]
        )

        pass
