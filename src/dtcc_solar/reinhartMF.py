import math
import numpy as np
from typing import Dict, Optional

from dtcc_solar.dome import Dome
from dtcc_core.model import Mesh
from dtcc_solar.logging import info


class ReinhartMF(Dome):
    """
    Parameterised Reinhart skydome (Tregenza basis, no ground by default).

    - Base is Tregenza 145 sky patches (8 bands, last is zenith).
    - Each patch except zenith is subdivided into M x M subpatches.
    - Zenith patch left whole (like your ReinhartM4).
    - Total patches (no ground): 1 + 144 * (M*M)

    Examples:
      ReinhartMF(4) -> 2305
      ReinhartMF(6) -> 5185
      ReinhartMF(8) -> 9217
    """

    def __init__(
        self, M: int, include_ground: bool = False, label: Optional[str] = None
    ):
        if M < 1:
            raise ValueError("M must be >= 1")

        self.M = int(M)
        self.include_ground = bool(include_ground)
        self.label = label or f"Reinhart MF:{self.M}"

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

    def expected_patch_count(self) -> int:
        # 145 sky patches total, 1 is zenith and not subdivided
        sky = 1 + 144 * (self.M * self.M)
        return sky + (1 if self.include_ground else 0)

    def create_mesh(self):
        dome_area = self.calc_hemisphere_area()
        elev_step = math.radians(12.0)

        for i in range(self.bands):
            patch_count = self.band_patches[i]
            elev1 = math.radians(self.elevs_deg[i])
            elev2 = min(elev1 + elev_step, math.radians(90.0))
            azim_step = 2 * math.pi / patch_count

            if elev1 < math.radians(84.0):
                # Subdivide each Tregenza patch into M x M subpatches
                for j in range(patch_count):
                    azim1 = j * azim_step
                    azim2 = (j + 1) * azim_step
                    self.subdivide_patch(
                        elev1,
                        elev2,
                        azim1,
                        azim2,
                        n_elev=self.M,
                        n_azim=self.M,
                        dome_area=dome_area,
                    )
            else:
                # Zenith patch (single, not subdivided)
                self.create_zenith_patch(elev1)

        # Optional: add a single ground direction (no geometry by default)
        # Use only if your matrix workflow expects it.
        if self.include_ground:
            self.ray_dirs.append([0.0, 0.0, -1.0])
            self.patch_zeniths.append(math.pi)  # pointing down
            self.solid_angles.append(2 * math.pi)  # entire lower hemisphere
            self.ray_areas.append(
                1.0
            )  # relative to "hemisphere area" this is arbitrary
            self.patch_counter += 1

        self.mesh = Mesh(vertices=np.array(self.vertices), faces=np.array(self.faces))

        tot_solid_angle = float(np.round(np.sum(self.solid_angles), 6))
        info("-----------------------------------------------------")
        info(f"{self.label} dome created:")
        info(
            f"  Number of patches: {self.patch_counter} (expected {self.expected_patch_count()})"
        )
        info(f"  Number of direction vectors: {len(self.ray_dirs)}")
        info(f"  Total solid angle: ~{tot_solid_angle} (sky-only expected ~6.283185)")
        info("-----------------------------------------------------")

    def subdivide_patch(self, elev1, elev2, azim1, azim2, n_elev, n_azim, dome_area):
        """
        Split a Tregenza patch (elev1–elev2, azim1–azim2) into n_elev × n_azim subpatches.
        """
        elev_edges = np.linspace(elev1, elev2, n_elev + 1)
        azim_edges = np.linspace(azim1, azim2, n_azim + 1)

        for ei in range(n_elev):
            for ai in range(n_azim):
                e1, e2 = float(elev_edges[ei]), float(elev_edges[ei + 1])
                a1, a2 = float(azim_edges[ai]), float(azim_edges[ai + 1])

                mid_elev = (e1 + e2) * 0.5
                mid_azim = (a1 + a2) * 0.5

                # geometry (quad -> 2 triangles)
                self.create_mesh_quad(a1, a2, e1, e2)

                # direction
                ray_dir = self.spherical_to_cartesian(mid_elev, mid_azim)
                self.ray_dirs.append(ray_dir)

                # zenith angle (angle from +z)
                self.patch_zeniths.append((math.pi / 2.0) - mid_elev)

                # solid angle and relative area
                solid_angle = self.solid_angle(e1, e2, a1, a2)
                self.solid_angles.append(solid_angle)

                patch_area = self.calc_sphere_patch_area(e1, e2, a1, a2)
                self.ray_areas.append(patch_area / dome_area)

                self.patch_counter += 1

    def create_zenith_patch(self, elev):
        """Single zenith patch like Tregenza (6 triangles fan)."""
        v_count = len(self.vertices)
        dome_area = self.calc_hemisphere_area()

        self.ray_dirs.append([0.0, 0.0, 1.0])
        self.patch_zeniths.append(np.pi / 2.0 - elev)

        solid_angle = self.calc_top_patch_solid_angle(elev)
        self.solid_angles.append(solid_angle)

        cap_area = self.calc_sphere_cap_area(elev)
        self.ray_areas.append(cap_area / dome_area)

        azimuth_step = 2 * math.pi / 6
        for i in range(6):
            azim = i * azimuth_step
            pt = self.spherical_to_cartesian(elev, azim)
            self.vertices.append(pt)

            idx1 = v_count + i
            idx2 = v_count + (i + 1) % 6
            idx3 = v_count + 6  # zenith point

            self.faces.append([idx1, idx2, idx3])

        self.vertices.append([0.0, 0.0, 1.0])  # zenith point
        self.patch_counter += 1

    def map_data_to_faces(self, data: np.ndarray) -> np.ndarray:
        """
        Map patch data to triangle face data for visualisation.
        Notes:
          - Each subpatch quad produces 2 triangles (so each patch value repeats twice)
          - Zenith patch is 6 triangles (repeat 6)
          - If include_ground=True: there is no geometry for ground by default, so it is ignored here.
        """
        expected = self.expected_patch_count()
        if len(data) != expected:
            raise ValueError(
                f"Data must have {expected} elements for {self.label} mapping."
            )

        data = np.asarray(data)
        if data.ndim == 2:
            data = np.sum(data, axis=1)

        # If ground is present, drop the last entry for face mapping (no ground geometry)
        if self.include_ground:
            data = data[:-1]

        # Zenith is last (in sky-only ordering)
        last = data[-1]

        # All patches except zenith are quads -> 2 triangles
        # count of non-zenith patches = (len(data)-1)
        tri_data = np.repeat(data[:-1], 2)

        # Zenith patch -> 6 triangles
        tri_data = np.append(tri_data, [last] * 6)

        return tri_data

    def map_dict_data_to_faces(self, dict_data: Dict) -> Dict:
        return {k: self.map_data_to_faces(v) for k, v in dict_data.items()}


# Convenience wrappers (if you still want named classes)
class ReinhartM6(ReinhartMF):
    def __init__(self, include_ground: bool = False):
        super().__init__(M=6, include_ground=include_ground, label="Reinhart MF:6")


class ReinhartM8(ReinhartMF):
    def __init__(self, include_ground: bool = False):
        super().__init__(M=8, include_ground=include_ground, label="Reinhart MF:8")
