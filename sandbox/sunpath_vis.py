import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar import utils
from dtcc_solar.utils import Vec3, Sun
from dtcc_solar.colors import *
from dtcc_solar.sunpath import SunpathUtils
from dtcc_solar.sunpath import Sunpath

from dtcc_model import Mesh, PointCloud

from typing import Dict, Any
from pprint import pp

# from timezonefinder import TimezoneFinder


class SunpathVis:
    def __init__(self):
        pass

    def initialise_plot(self, r, title):
        plt.rcParams["figure.figsize"] = (16, 11)
        plt.title(label=title, fontsize=44, color="black")
        ax = plt.axes(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(-r, r)
        return ax

    def plot_imported_sunpath_diagarm(self, pts, radius, ax, cmap):
        n = int(0.5 * len(pts[0]))
        x, y, z = [], [], []
        for hour in pts:
            for i in range(n):
                pt = pts[hour][i]
                x.append(pt.x)
                y.append(pt.y)
                z.append(pt.z)

        ax.scatter3D(x, y, z, c=z, cmap=cmap, vmin=0, vmax=radius)

    def plot_analemmas(
        self, all_sun_pos: Dict[int, list[Vec3]], radius, ax, plot_night, cmap, gmt_diff
    ):
        x, y, z = [], [], []
        z_max_indices = []
        counter = 0

        for hour in all_sun_pos:
            z_max = -100000000
            z_max_index = -1
            for i in range(len(all_sun_pos[hour])):
                x.append(all_sun_pos[hour][i].x)
                y.append(all_sun_pos[hour][i].y)
                z.append(all_sun_pos[hour][i].z)
                if all_sun_pos[hour][i].z > z_max:
                    z_max = all_sun_pos[hour][i].z
                    z_max_index = counter
                counter += 1
            z_max_indices.append(z_max_index)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        day_indices = np.where(z > 0)
        night_indices = np.where(z <= 0)

        ax.scatter3D(
            x[day_indices],
            y[day_indices],
            z[day_indices],
            c=z[day_indices],
            cmap=cmap,
            vmin=0,
            vmax=radius,
        )
        if plot_night:
            ax.scatter3D(
                x[night_indices], y[night_indices], z[night_indices], color="w"
            )

        # Add label to the hourly loops
        for i in range(len(z_max_indices)):
            utc_hour = i
            local_hour = SunpathUtils.convert_utc_to_local_time(utc_hour, gmt_diff)
            text_pos_max = np.array(
                [x[z_max_indices[i]], y[z_max_indices[i]], z[z_max_indices[i]]]
            )
            ax.text(
                text_pos_max[0],
                text_pos_max[1],
                text_pos_max[2],
                str(local_hour),
                fontsize=12,
            )

    def plot_daypath(self, sun_pos_dict, radius, ax, plot_night):
        sun_pos_arr = utils.dict_2_np_array(sun_pos_dict)
        day_indices = np.where(sun_pos_arr[:, 2] > 0)
        night_indices = np.where(sun_pos_arr[:, 2] <= 0)
        z_color = sun_pos_arr[day_indices, 2]

        ax.scatter3D(
            sun_pos_arr[day_indices, 0],
            sun_pos_arr[day_indices, 1],
            sun_pos_arr[day_indices, 2],
            c=z_color,
            cmap="autumn_r",
            vmin=0,
            vmax=radius,
        )
        if plot_night:
            ax.scatter3D(
                sun_pos_arr[night_indices, 0],
                sun_pos_arr[night_indices, 1],
                sun_pos_arr[night_indices, 2],
                color="w",
            )

    def plot_single_sun(self, x, y, z, radius, ax):
        z_color = z
        ax.scatter3D(x, y, z, c=z_color, cmap="autumn_r", vmin=0, vmax=radius)

    def plot_multiple_suns(self, sun_pos, radius, ax, plot_night):
        z = sun_pos[:, 2]
        day_indices = np.where(z > 0)
        night_indices = np.where(z <= 0)
        zColor = z[day_indices]
        ax.scatter3D(
            sun_pos[day_indices, 0],
            sun_pos[day_indices, 1],
            sun_pos[day_indices, 2],
            c=zColor,
            cmap="autumn_r",
            vmin=0,
            vmax=radius,
        )
        if plot_night:
            ax.scatter3D(
                sun_pos[night_indices, 0],
                sun_pos[night_indices, 1],
                sun_pos[night_indices, 2],
                color="w",
            )
