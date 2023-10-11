import numpy as np
import pandas as pd
from dtcc_model import Mesh, PointCloud
from dtcc_solar.utils import SunQuad
from dtcc_solar.sunpath import Sunpath
from shapely import LineString
from pprint import pp
from dtcc_solar.utils import distance, normalise_vector
from dtcc_solar.logging import info, debug, warning, error


class SkyCylinder:
    radius: float
    div_tangent: int
    div_height: int
    center: np.ndarray
    mesh: Mesh
    pc: PointCloud
    quads: list[SunQuad]
    quad_mid_pts: np.ndarray  # [n_quads * 3] mid points for all quads

    def __init__(self, sunpath: Sunpath, horizon_z: float, div_n: int, div_m: int):
        self.center = np.array([0, 0, 0])
        self.create_skycylinder_mesh(sunpath, horizon_z, div_n, div_m)

    def create_skycylinder_mesh(
        self, sunpath: Sunpath, horizon_z: float, div_n: int, div_m: int
    ):
        """Creates a cylindrical mesh mapped to the sphere of the sunpath diagram"""
        day_loop1, day_loop2 = self._calc_outermost_day_loops(sunpath)

        self.pc, self.mesh, self.quads = self._create_mesh(
            sunpath.radius,
            day_loop1,
            day_loop2,
            div_n,
            div_m,
        )

        self._process_quads(horizon_z)

    def _calc_outermost_day_loops(self, sunpath: Sunpath):
        """Returns the outermost day path loops"""
        dates = pd.date_range(start="2019-01-01", end="2019-12-31", freq="1D")
        sun_pos_dict = sunpath.get_daypaths(dates, 10)
        day_loops = []
        avrg_pts = []

        for day in sun_pos_dict:
            day_points = []
            for sun in sun_pos_dict[day]:
                day_points.append(np.array([sun.x, sun.y, sun.z]))

            day_points = np.array(day_points)
            day_loop = LineString(day_points)
            day_loops.append(day_loop)

            avrg_pt = np.mean(day_points, axis=0)
            avrg_pts.append(avrg_pt)

        avrg_pts = np.array(avrg_pts)

        # Find the index of the rings that are the furtherst from the center.
        index1 = self._get_pt_index_far_from_other_pt(avrg_pts, np.array([0, 0, 0]))

        # Find the index furtherst from the other ring
        index2 = self._get_pt_index_far_from_other_pt(avrg_pts, avrg_pts[index1, :])

        return day_loops[index1], day_loops[index2]

    def _get_pt_index_far_from_other_pt(self, pts: np.ndarray, far_from_pt: np.ndarray):
        """Returns the index of the point that is the furtherst from the far_from_pt"""
        dmax = -10000000000
        index = None
        for i, pt in enumerate(pts):
            d = distance(pt, far_from_pt)
            if d > dmax:
                dmax = d
                index = i

        return index

    def _create_mesh(
        self, radius: float, loop_1: LineString, loop_2: LineString, n: int, m: int
    ):
        """Creates a sunpath diagram mesh inbetween the outermost day path loops"""
        points = []
        avrg_length = (loop_1.length + loop_2.length) / 2
        step_n = avrg_length / n
        ds_n = np.arange(0, avrg_length, step_n)
        faces = []
        quads = []
        face_count = 0
        quad_count = 0

        for i, d_n in enumerate(ds_n):
            pt1 = loop_1.interpolate(d_n)
            pt2 = loop_2.interpolate(d_n)

            line = LineString(np.array([pt1, pt2]))
            line_length = line.length

            ds_m = np.linspace(0, line_length, m)

            for j, d_m in enumerate(ds_m):
                pt = line.interpolate(d_m)
                pt = radius * normalise_vector(np.array([pt.x, pt.y, pt.z]))

                if i < (n - 1):
                    if j < (m - 1):
                        current = (m * i) + j
                        face1 = [current, current + 1, current + m]
                        face2 = [current + m, current + 1, current + m + 1]
                        faces.append(face1)
                        faces.append(face2)
                        quads.append(SunQuad(face_count, face_count + 1, quad_count))
                        quad_count += 1
                        face_count += 2

                elif i == (n - 1):
                    if j < (m - 1):
                        current = (m * i) + j
                        face1 = [current, current + 1, j]
                        face2 = [j, current + 1, j + 1]
                        faces.append(face1)
                        faces.append(face2)
                        quads.append(SunQuad(face_count, face_count + 1, quad_count))
                        quad_count += 1
                        face_count += 2

                points.append(pt)

        points = np.array(points)
        pc = PointCloud(points=points)
        faces = np.array(faces)
        mesh = Mesh(vertices=points, faces=faces)

        return pc, mesh, quads

    def _process_quads(self, horizon_z: float):
        """Calculates the center and area of each quad"""
        for sun_quad in self.quads:
            face_a = self.mesh.faces[sun_quad.face_index_a]
            face_b = self.mesh.faces[sun_quad.face_index_b]

            v1 = self.mesh.vertices[face_a[0]]
            v2 = self.mesh.vertices[face_a[1]]
            v3 = self.mesh.vertices[face_b[2]]

            v4 = self.mesh.vertices[face_b[0]]
            v5 = self.mesh.vertices[face_b[1]]
            v6 = self.mesh.vertices[face_b[2]]

            # Picking perimeter verices without duplicates
            sun_quad.center = (v1 + v2 + v4 + v6) / 4.0

            if sun_quad.center[2] > horizon_z:
                sun_quad.over_horizon = True

            vec1 = v2 - v1
            vec2 = v3 - v1
            area_a = 0.5 * np.cross(vec1, vec2)

            vec3 = v5 - v6
            vec4 = v4 - v6
            area_b = 0.5 * np.cross(vec3, vec4)

            sun_quad.area = area_a + area_b
