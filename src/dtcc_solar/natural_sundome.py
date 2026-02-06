import math
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod
from dtcc_solar.dome import Dome
from dtcc_core.model import Mesh
from dtcc_solar.logging import info, debug, warning, error
from dtcc_solar.sunpath import Sunpath


class NaturalSunDome(Dome):

    def __init__(self, sunpath: Sunpath):
        self.ray_dirs = []
        self.ray_areas = []
        self.solid_angles = []

        self.patch_counter = 0
        self.bands = 15
        self.create_mesh(sunpath)

    def create_mesh(self, sunpath: Sunpath):
        n_suns = len(sunpath.sunc.sun_vecs)
        sa_per_sun = (2.0 * math.pi) / n_suns  # equal solid angle per sun
        tot_solid_angle = 0
        for sun_vec in sunpath.sunc.sun_vecs:
            self.solid_angles.append(1.0)
            self.ray_dirs.append(sun_vec)
            self.patch_counter += 1
            tot_solid_angle += sa_per_sun

        info("-----------------------------------------------------")
        info(f"Sun dome with natural sun positions created:")
        info(f"  Number of patches: {self.patch_counter}")
        info(f"  Number of direction vectors: {len(self.ray_dirs)}")
        info(f"  Total solid angle: ~{tot_solid_angle}, expected: ~6.283185")
        info("-----------------------------------------------------")

    def create_zenith_patch(self, elev1, elev2):
        pass

    def map_data_to_faces(self, data: np.ndarray) -> np.ndarray:
        pass

    def map_dict_data_to_faces(self, dict_data: Dict) -> np.ndarray:
        pass

    def create_zenith_mesh(self, index, v_count, elev1, elev2, azim1, azim2):

        pass
