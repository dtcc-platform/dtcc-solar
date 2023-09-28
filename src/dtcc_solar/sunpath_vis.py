import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar import utils
from dtcc_solar.utils import Vec3, Sun
from dtcc_solar.viewer import Colors
from dtcc_solar.sunpath import SunpathUtils
from dtcc_solar.sunpath import Sunpath

from dtcc_model import Mesh, PointCloud

from typing import Dict, Any

# from timezonefinder import TimezoneFinder


class SunpathMesh:
    radius: float
    origin: np.ndarray
    sun_meshes: list[Any]
    analemmas_meshes: list[Any]
    daypath_meshes: list[Any]
    pc: PointCloud

    def __init__(self, radius: float):
        self.radius = radius
        self.origin = np.array([0, 0, 0])

    def get_analemmas_meshes(self):
        return self.analemmas_meshes

    def get_analemmas_pc(self):
        return self.pc

    def get_daypath_meshes(self):
        return self.daypath_meshes

    def get_sun_meshes(self):
        return self.sun_meshes

    def create_sunpath_diagram(
        self, suns: list[Sun], sunpath: Sunpath, city_model: SolarEngine, colors: Colors
    ):
        # Create analemmas mesh
        [sunX, sunY, sunZ, analemmas_dict] = sunpath.get_analemmas(2019, 5)
        self.analemmas_meshes = self.create_sunpath_loops(
            sunX, sunY, sunZ, city_model.sunpath_radius, colors
        )

        # Create mesh for day path
        [sunX, sunY, sunZ] = sunpath.get_daypaths(
            pd.to_datetime(["2019-06-21", "2019-03-21", "2019-12-21"]), 10
        )
        self.daypath_meshes = self.create_sunpath_loops(
            sunX, sunY, sunZ, city_model.sunpath_radius, colors
        )

        self.sun_meshes = self.create_solar_spheres(suns, city_model.sun_size)

    def create_sunpath_diagram_gl(
        self,
        suns: list[Sun],
        sunpath: Sunpath,
        solar_engine: SolarEngine,
        colors: Colors,
    ):
        r = solar_engine.sunpath_radius

        [sunX, sunY, sunZ, analemmas_dict] = sunpath.get_analemmas(2019, 2)

        self.analemmas_meshes = self.create_sunpath_loops_mesh(
            sunX, sunY, sunZ, r, 3.0, colors
        )

        self.pc = self.create_sunpath_pc(sunX, sunY, sunZ, r, colors)

        [sunX, sunY, sunZ] = sunpath.get_daypaths(
            pd.to_datetime(["2019-06-21", "2019-03-21", "2019-12-21"]), 2
        )

        self.daypath_meshes = self.create_sunpath_loops_mesh(
            sunX, sunY, sunZ, r, 3.0, colors
        )

    def create_solar_sphere(self, sunPos, sunSize):
        sunMesh = trimesh.primitives.Sphere(
            radius=sunSize, center=sunPos, subdivisions=4
        )
        sunMesh.visual.face_colors = [1.0, 0.5, 0, 1.0]
        return sunMesh

    def create_solar_spheres(self, suns: list[Sun], sunSize):
        sunMeshes = []
        for i in range(0, len(suns)):
            if suns[i].over_horizon:
                sun_pos = utils.convert_vec3_to_ndarray(suns[i].position)
                sunMesh = trimesh.primitives.Sphere(
                    radius=sunSize, center=sun_pos, subdivisions=1
                )
                sunMesh.visual.face_colors = [1.0, 0.5, 0, 1.0]
                sunMeshes.append(sunMesh)
        return sunMeshes

    def create_sunpath_loops(self, x, y, z, radius, colors: Colors):
        path_meshes = []
        for h in x:
            vs = np.zeros((len(x[h]) + 1, 3))
            vi = np.zeros((len(x[h])), dtype=int)
            lines = []
            path_colors = []

            for i in range(0, len(x[h])):
                sunPos = [x[h][i], y[h][i], z[h][i]]
                vs[i, :] = sunPos
                vi[i] = i
                index2 = i + 1
                color = colors.get_blended_color_yellow_red(radius, z[h][i])
                path_colors.append(color)
                line = trimesh.path.entities.Line([i, index2])
                lines.append(line)

            vs[len(x[h]), :] = vs[0, :]

            path = trimesh.path.Path3D(entities=lines, vertices=vs, colors=path_colors)

            path_meshes.append(path)

        return path_meshes

    def create_sunpath_loops_mesh(
        self, x, y, z, radius: float, width: float, colors: Colors
    ):
        meshes = []

        for h in x:
            n_suns = len(x[h])
            offset_vertices_L = np.zeros((n_suns + 1, 3))
            offset_vertices_R = np.zeros((n_suns + 1, 3))

            v_counter = 0
            vertices = []
            faces = []
            vertex_colors = []

            for i in range(0, n_suns - 1):
                i_next = i + 1
                if i == n_suns - 1:
                    i_next = 0

                sun_pos_1 = np.array([x[h][i], y[h][i], z[h][i]])
                sun_pos_2 = np.array([x[h][i_next], y[h][i_next], z[h][i_next]])
                vec_1 = utils.normalise_vector(0.5 * (sun_pos_1 + sun_pos_2))
                vec_2 = utils.normalise_vector(sun_pos_2 - sun_pos_1)
                vec_3 = utils.cross_product(vec_2, vec_1)

                # Offset vertices to create a width for path
                offset_vertices_L[i, :] = sun_pos_1 + (width * vec_3)
                offset_vertices_R[i, :] = sun_pos_1 - (width * vec_3)

                vertices.append(offset_vertices_L[i, :])
                vertices.append(offset_vertices_R[i, :])

                if i < (n_suns - 2):
                    faces.append([v_counter, v_counter + 2, v_counter + 1])
                    faces.append([v_counter + 1, v_counter + 2, v_counter + 3])
                    v_counter += 2
                else:
                    faces.append([v_counter, 0, v_counter + 1])
                    faces.append([v_counter + 1, 0, 1])

                v_color = colors.get_blended_color_yellow_red(radius, z[h][i])

                vertex_colors.append(v_color)
                vertex_colors.append(v_color)

            vertices = np.array(vertices)
            faces = np.array(faces)
            vertex_colors = np.array(vertex_colors)
            mesh = Mesh(vertices=vertices, vertex_colors=vertex_colors, faces=faces)
            meshes.append(mesh)

        return meshes

    def create_sunpath_pc(self, x, y, z, radius: float, colors: Colors):
        points = []
        for h in x:
            n_suns = len(x[h])

            for i in range(0, n_suns - 1):
                sun_pos_1 = np.array([x[h][i], y[h][i], z[h][i]])
                points.append(sun_pos_1)

        points = np.array(points)
        print(points)
        pc = PointCloud(points=points)
        return pc


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

    def plot_daypath(self, x_dict, y_dict, z_dict, radius, ax, plot_night):
        for key in x_dict:
            x = x_dict[key]
            y = y_dict[key]
            z = z_dict[key]

            day_indices = np.where(z > 0)
            night_indices = np.where(z <= 0)
            z_color = z[day_indices]
            ax.scatter3D(
                x[day_indices],
                y[day_indices],
                z[day_indices],
                c=z_color,
                cmap="autumn_r",
                vmin=0,
                vmax=radius,
            )
            if plot_night:
                ax.scatter3D(
                    x[night_indices], y[night_indices], z[night_indices], color="w"
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
