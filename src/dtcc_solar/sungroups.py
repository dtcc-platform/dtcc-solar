import numpy as np
import pandas as pd
from dtcc_model import Mesh, PointCloud
from dtcc_solar.utils import SunCollection
from pprint import pp
from dtcc_solar.utils import distance
from dtcc_solar.logging import info, debug, warning, error
from dtcc_viewer.opengl_viewer import Window, Scene
from dtcc_model import Mesh, PointCloud


class SunGroups:
    centers: np.ndarray
    sun_indices: np.ndarray
    centers_pc: PointCloud
    has_sun: np.ndarray

    dict_centers: dict
    dict_radius: dict

    list_centers: list

    def __init__(self, sun_pos_dict: dict, sunc: SunCollection):
        self._create_group_centers(sun_pos_dict)
        self._match_suns_and_quads(sunc)
        self._mask_data()

    def _view(self):
        pc = PointCloud(points=self.list_centers)
        window = Window(1200, 800)
        scene = Scene()
        scene.add_pointcloud(pc=pc, name="Sun centers", size=1 * self.w)
        window.render(scene)

    def _create_group_centers(self, sun_pos_dict):
        self.dict_centers = dict.fromkeys(np.arange(0, 24), [])
        self.dict_radius = dict.fromkeys(np.arange(0, 24), 0.0)
        self.dict_indices = dict.fromkeys(np.arange(0, 24), [])
        self.list_centers = []
        self.sun_indices = []

        center_index = 0
        for h in sun_pos_dict:
            n_suns = len(sun_pos_dict[h])
            pos = sun_pos_dict[h]

            for i in range(0, n_suns):
                sun_pos_prev = np.array([pos[i - 1].x, pos[i - 1].y, pos[i - 1].z])
                sun_pos = np.array([pos[i].x, pos[i].y, pos[i].z])

                if i == n_suns - 1:
                    sun_pos_next = np.array([pos[0].x, pos[0].y, pos[0].z])
                else:
                    sun_pos_next = np.array([pos[i + 1].x, pos[i + 1].y, pos[i + 1].z])

                d1 = distance(sun_pos_prev, sun_pos)
                d2 = distance(sun_pos, sun_pos_next)

                d = (d1 + d2) / 2.0

                self.list_centers.append(sun_pos)
                self.dict_radius[h] = d
                self.dict_centers[h].append(sun_pos)
                self.dict_indices[h].append(center_index)
                self.sun_indices.append([])

                center_index += 1

        self.has_sun = np.zeros(len(self.list_centers), dtype=bool)
        self.list_centers = np.array(self.list_centers)

    def _match_suns_and_quads(self, sunc: SunCollection):
        for sun_index, sun_pos in enumerate(sunc.positions):
            dmin = 1000000000
            centre_index = None
            ts = sunc.time_stamps[sun_index]
            h = ts.hour
            # Use the hour to reduce the seach domain for the closest group center
            for j, pos in enumerate(self.dict_centers[h]):
                d = distance(pos, sun_pos)
                if d < dmin:
                    dmin = d
                    centre_index = self.dict_indices[h][j]

            self.sun_indices[centre_index].append(sun_index)
            self.has_sun[centre_index] = True

    def _mask_data(self):
        new_sun_indices = []
        new_list_centers = []
        suns_per_group = 0
        counter = 0
        for i, indices in enumerate(self.sun_indices):
            if self.has_sun[i]:
                new_sun_indices.append(indices)
                new_list_centers.append(self.list_centers[i])
                suns_per_group += len(indices)
                counter += 1

        suns_per_group /= counter
        self.sun_indices = new_sun_indices
        self.list_centers = np.array(new_list_centers)

        self.centers_pc = PointCloud(points=self.list_centers)
        info(f"The average number of suns per group is: {suns_per_group}")
