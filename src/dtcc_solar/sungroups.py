import numpy as np
import pandas as pd
from dtcc_model import Mesh, PointCloud
from dtcc_solar.utils import SunCollection
from pprint import pp
from dtcc_solar.utils import distance
from dtcc_solar.logging import info, debug, warning, error
from dtcc_model import Mesh, PointCloud


class SunGroups:
    """
    Class for grouping sun positions based on count along the analemmas.

    Attributes
    ----------
    centers : np.ndarray
        Array of center positions of sun groups.
    sun_indices : np.ndarray
        Array of sun indices corresponding to the groups.
    centers_pc : PointCloud
        Point cloud of the centers of sun groups.
    has_sun : np.ndarray
        Boolean array indicating whether each group has a sun.
    dict_centers : dict
        Dictionary of centers of sun groups indexed by hour.
    dict_radius : dict
        Dictionary of radii of sun groups indexed by hour.
    list_centers : list
        List of center positions of sun groups.
    dict_indices : dict
        Dictionary of indices of sun group centers indexed by hour.
    """

    centers: np.ndarray
    sun_indices: np.ndarray
    centers_pc: PointCloud
    has_sun: np.ndarray
    dict_centers: dict
    dict_radius: dict
    list_centers: list

    def __init__(self, sun_pos_dict: dict, sunc: SunCollection):
        """
        Initialize the SunGroups object with sun positions and sun collection.

        Parameters
        ----------
        sun_pos_dict : dict
            Dictionary containing positions for group centers indexed by hour.
        sunc : SunCollection
            Collection of sun data including positions and timestamps.
        """
        self._create_group_centers(sun_pos_dict)
        self._match_suns_and_group_centers(sunc)
        self._clean_data()

    def _create_group_centers(self, sun_pos_dict):
        """
        Create the group centers from the given sun positions.

        Parameters
        ----------
        sun_pos_dict : dict
            Dictionary containing sun positions indexed by hour.
        """
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

    def _match_suns_and_group_centers(self, sunc: SunCollection):
        """
        Match the suns with the nearest group centers.

        Parameters
        ----------
        sunc : SunCollection
            Collection of sun data including positions and timestamps.
        """
        for sun_index, sun_pos in enumerate(sunc.positions):
            dmin = 1000000000
            centre_index = None
            ts = sunc.time_stamps[sun_index]
            h = ts.hour
            # Using the hour to reduce the seach domain for the closest group center
            for j, pos in enumerate(self.dict_centers[h]):
                d = distance(pos, sun_pos)
                if d < dmin:
                    dmin = d
                    centre_index = self.dict_indices[h][j]

            self.sun_indices[centre_index].append(sun_index)
            self.has_sun[centre_index] = True

    def _clean_data(self):
        """
        Mask the data to filter out empty sun groups and update the list of centers.
        """
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
