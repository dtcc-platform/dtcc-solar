import math
import numpy as np
from typing import Dict
from dtcc_solar.skydome import Skydome
from dtcc_core.model import Mesh
from dtcc_solar.logging import info


class ReinhartM4(Skydome):
    """
    Reinhart MF:4 skydome (2305 patches, no ground).
    - 145 Tregenza patches
    - Each patch (except zenith) split into 16 subpatches
    - Zenith patch left whole
    """

    def __init__(self):
        self.vertices = []
        self.faces = []
        self.ray_dirs = []
        self.patch_zeniths = []
        self.ray_areas = []
        self.solid_angles = []
        self.quad_midpoints = []
        self.patch_counter = 0

        # Tregenza band definitions
        self.bands = 8
        self.band_patches = [30, 30, 24, 24, 18, 12, 6, 1]
        self.elevs_deg = [0, 12, 24, 36, 48, 60, 72, 84]

        self.create_mesh()

    def create_mesh(self):
        dome_area = self.calc_hemisphere_area()
        elev_step = math.radians(12.0)

        for i in range(self.bands):
            patch_count = self.band_patches[i]
            elev1 = math.radians(self.elevs_deg[i])
            elev2 = elev1 + elev_step
            azim_step = 2 * math.pi / patch_count

            if elev1 < math.radians(84.0):
                # Subdivide each Tregenza patch into 16 subpatches (4x4 grid)
                for j in range(patch_count):
                    azim1 = j * azim_step
                    azim2 = (j + 1) * azim_step
                    self.subdivide_patch(elev1, elev2, azim1, azim2, 4, 4, dome_area)
            else:
                # Zenith patch is not subdivided
                self.create_zenith_patch(elev1)

        self.mesh = Mesh(vertices=np.array(self.vertices), faces=np.array(self.faces))

        tot_solid_angle = np.round(np.sum(self.solid_angles), 6)
        info("-----------------------------------------------------")
        info(f"Reinhart MF:4 skydome created:")
        info(f"  Number of patches: {self.patch_counter} (expected 2305)")
        info(f"  Number of direction vectors: {len(self.ray_dirs)}")
        info(f"  Total solid angle: ~{tot_solid_angle}, expected: ~6.283185")
        info("-----------------------------------------------------")

    def subdivide_patch(self, elev1, elev2, azim1, azim2, n_elev, n_azim, dome_area):
        """
        Split a Tregenza patch (defined by elev1–elev2, azim1–azim2)
        into a grid of n_elev × n_azim subpatches.
        """
        elev_edges = np.linspace(elev1, elev2, n_elev + 1)
        azim_edges = np.linspace(azim1, azim2, n_azim + 1)

        for ei in range(n_elev):
            for ai in range(n_azim):
                e1, e2 = elev_edges[ei], elev_edges[ei + 1]
                a1, a2 = azim_edges[ai], azim_edges[ai + 1]

                mid_elev = (e1 + e2) / 2.0
                mid_azim = (a1 + a2) / 2.0

                # Build geometry
                self.create_mesh_quad(a1, a2, e1, e2)

                # Midpoint direction
                ray_dir = self.spherical_to_cartesian(mid_elev, mid_azim)
                self.ray_dirs.append(ray_dir)

                # Zenith angle
                self.patch_zeniths.append((math.pi / 2.0) - mid_elev)

                # Solid angle and relative area
                solid_angle = self.solid_angle(e1, e2, a1, a2)
                self.solid_angles.append(solid_angle)

                patch_area = self.calc_sphere_patch_area(e1, e2, a1, a2)
                self.ray_areas.append(patch_area / dome_area)

                self.patch_counter += 1

    def create_zenith_patch(self, elev):
        """Single zenith patch like Tregenza."""
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
        if len(data) != 2305:
            raise ValueError("Data must have 2305 elements for Reinhart MF:4 mapping.")

        data = np.array(data)
        if len(data.shape) == 2:
            data = np.sum(data, axis=1)

        last = data[-1]

        # All patches except the last have 2 triangles (quads → 2 tris)
        data = np.repeat(data[0:2304], 2)

        # The last patch (zenith) has 6 triangles
        data = np.append(data, [last, last, last, last, last, last])

        return data

    def map_dict_data_to_faces(self, dict_data: Dict) -> np.ndarray:
        mapped_dict = {}
        for key, data in dict_data.items():
            mapped_dict[key] = self.map_data_to_faces(data)
        return mapped_dict
