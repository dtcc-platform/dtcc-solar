import numpy as np
import pandas as pd
from dtcc_model import Mesh, PointCloud
from dtcc_solar.utils import SunCollection
from shapely import LineString
from pprint import pp
from dtcc_solar.utils import distance, unitize
from dtcc_solar.logging import info, debug, warning, error
from dataclasses import dataclass, field


@dataclass
class SunQuad:
    face_index_a: int
    face_index_b: int
    id: int
    area: float = 0.0
    has_sun: bool = False
    over_horizon = False
    center: np.ndarray = field(default_factory=lambda: np.empty(0))
    sun_indices: list[int] = field(default_factory=list)
    mesh: Mesh = None


class SunQuads:
    """
    Class for creating a mesh representation of the entire a sun path diagram. The quads
    in the mesh are then used to group sun positions for faster analysis.


    Attributes
    ----------
    radius : float
        Radius of the sun dome.
    div_tangent : int
        Number of divisions along the tangent.
    div_height : int
        Number of divisions along the height.
    center : np.ndarray
        Center of the sun dome.
    mesh : Mesh
        Mesh representing the sun dome.
    submesh : Mesh
        Submesh containing only active quads.
    pc : PointCloud
        Point cloud of vertices in the sun dome mesh.
    all_quads : list[SunQuad]
        List of all quads in the sun dome mesh.
    active_quads : list[SunQuad]
        List of active quads with suns.
    quad_mid_pts : np.ndarray
        Midpoints of all quads in the sun dome mesh.
    """

    radius: float
    div_tangent: int
    div_height: int
    center: np.ndarray
    mesh: Mesh
    submesh: Mesh
    pc: PointCloud
    all_quads: list[SunQuad]
    active_quads: list[SunQuad]
    quad_mid_pts: np.ndarray  # [n_quads * 3] mid points for all quads

    def __init__(self, sun_pos_dict: dict, radius: float, div: tuple[int, int]):
        """
        Initialize the SunDome object with sun positions, radius, and divisions.

        Parameters
        ----------
        sun_pos_dict : dict
            Dictionary containing sun positions indexed by day.
        radius : float
            Radius of the sun dome.
        div : tuple[int, int]
            Number of divisions along the tangent and height.
        """
        self.center = np.array([0, 0, 0])
        self.all_quads = []
        self.active_quads = []
        day_loop1, day_loop2 = self._calc_outermost_day_loops(sun_pos_dict)
        self._create_mesh(radius, day_loop1, day_loop2, div[0], div[1])
        self._process_quads()

    def _calc_outermost_day_loops(self, sun_pos_dict: dict):
        """
        Returns the outermost day path loops.

        Parameters
        ----------
        sun_pos_dict : dict
            Dictionary containing sun positions indexed by day.

        Returns
        -------
        tuple[LineString, LineString]
            The outermost day loops.
        """
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
        """
        Returns the index of the point that is the farthest from the given point.

        Parameters
        ----------
        pts : np.ndarray
            Array of points.
        far_from_pt : np.ndarray
            Point to compare the distance from.

        Returns
        -------
        int
            Index of the farthest point.
        """
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
        """
        Creates a sun path diagram mesh between the outermost day path loops.

        Parameters
        ----------
        radius : float
            Radius of the sun dome.
        loop_1 : LineString
            First outermost day path loop.
        loop_2 : LineString
            Second outermost day path loop.
        n : int
            Number of divisions along the tangent.
        m : int
            Number of divisions along the height.
        """
        points = []
        avrg_length = (loop_1.length + loop_2.length) / 2
        step_n = avrg_length / n
        ds_n = np.arange(0, avrg_length, step_n)
        faces = []
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
                pt = radius * unitize(np.array([pt.x, pt.y, pt.z]))

                if i < (n - 1):
                    if j < (m - 1):
                        current = (m * i) + j
                        face1 = [current, current + 1, current + m]
                        face2 = [current + m, current + 1, current + m + 1]
                        faces.append(face1)
                        faces.append(face2)
                        sun_quad = SunQuad(face_count, face_count + 1, quad_count)
                        self.all_quads.append(sun_quad)
                        quad_count += 1
                        face_count += 2

                elif i == (n - 1):
                    if j < (m - 1):
                        current = (m * i) + j
                        face1 = [current, current + 1, j]
                        face2 = [j, current + 1, j + 1]
                        faces.append(face1)
                        faces.append(face2)
                        sun_quad = SunQuad(face_count, face_count + 1, quad_count)
                        self.all_quads.append(sun_quad)
                        quad_count += 1
                        face_count += 2

                points.append(pt)

        points = np.array(points)
        self.pc = PointCloud(points=points)
        faces = np.array(faces)
        self.mesh = Mesh(vertices=points, faces=faces)

    def _process_quads(self):
        """Calculates the center and area of each quad."""
        for sun_quad in self.all_quads:
            face_a = self.mesh.faces[sun_quad.face_index_a]
            face_b = self.mesh.faces[sun_quad.face_index_b]

            v1 = self.mesh.vertices[face_a[0]]
            v2 = self.mesh.vertices[face_a[1]]
            v3 = self.mesh.vertices[face_b[2]]

            v4 = self.mesh.vertices[face_b[0]]
            v5 = self.mesh.vertices[face_b[1]]
            v6 = self.mesh.vertices[face_b[2]]

            sun_quad.mesh = self._create_sun_quad_mesh(face_a, face_b)

            # Picking perimeter verices without duplicates
            sun_quad.center = (v1 + v2 + v4 + v6) / 4.0

            vec1 = v2 - v1
            vec2 = v3 - v1
            area_a = 0.5 * np.cross(vec1, vec2)

            vec3 = v5 - v6
            vec4 = v4 - v6
            area_b = 0.5 * np.cross(vec3, vec4)

            sun_quad.area = area_a + area_b

    def calc_sub_sundome_mesh(self, tolerance: float):
        """
        Returns a sub sun dome mesh with the quads that have suns.

        Parameters
        ----------
        tolerance : float
            Tolerance for vertex comparison.
        """
        new_faces = []
        new_vertices = []
        v_index = 0
        for quad in self.all_quads:
            if quad.has_sun:
                face1 = self.mesh.faces[quad.face_index_a, :]
                face2 = self.mesh.faces[quad.face_index_b, :]

                vertices1 = self.mesh.vertices[face1, :]
                vertices2 = self.mesh.vertices[face2, :]
                new_face1 = []
                for v in vertices1:
                    existing_v_index = self._vertex_exists(new_vertices, v, tolerance)
                    if existing_v_index == -1:
                        # Vertex did not exist so a new one is created
                        new_vertices.append(v)
                        new_face1.append(v_index)
                        v_index += 1
                    else:
                        new_face1.append(existing_v_index)

                new_face2 = []
                for v in vertices2:
                    existing_v_index = self._vertex_exists(new_vertices, v, tolerance)
                    if existing_v_index == -1:
                        # Vertex did not exist so a new one is created
                        new_vertices.append(v)
                        new_face2.append(v_index)
                        v_index += 1
                    else:
                        new_face2.append(existing_v_index)

                new_faces.append(new_face1)
                new_faces.append(new_face2)

        new_vertices = np.array(new_vertices)
        new_faces = np.array(new_faces)
        mesh = Mesh(vertices=new_vertices, faces=new_faces)

        self.submesh = mesh

    def _vertex_exists(self, vertices, vertex, tolerance):
        """
        Checks if a vertex exists within a given tolerance.

        Parameters
        ----------
        vertices : list
            List of existing vertices.
        vertex : np.ndarray
            Vertex to check.
        tolerance : float
            Tolerance for distance comparison.

        Returns
        -------
        int
            Index of the existing vertex or -1 if not found.
        """
        for i, v in enumerate(vertices):
            d = distance(v, vertex)
            if d < tolerance:
                return i

        return -1

    def _create_sun_quad_mesh(self, face1, face2):
        """
        Creates a mesh for a sun quad.

        Parameters
        ----------
        face1 : list[int]
            Indices of the first face vertices.
        face2 : list[int]
            Indices of the second face vertices.

        Returns
        -------
        Mesh
            Mesh representing the sun quad.
        """
        v1 = self.mesh.vertices[face1[0]]
        v2 = self.mesh.vertices[face1[1]]
        v3 = self.mesh.vertices[face1[2]]

        v4 = self.mesh.vertices[face2[0]]
        v5 = self.mesh.vertices[face2[1]]
        v6 = self.mesh.vertices[face2[2]]

        quad_vertices = np.array([v1, v2, v3, v4, v5, v6])
        quad_faces = np.array([[0, 1, 2], [3, 4, 5]])

        quad_mesh = Mesh(vertices=quad_vertices, faces=quad_faces)

        return quad_mesh

    def match_suns_and_quads(self, sunc: SunCollection):
        """
        Matches the suns with the nearest quads.

        Parameters
        ----------
        sunc : SunCollection
            Collection of sun data including positions and timestamps.
        """
        for i, sun_pos in enumerate(sunc.positions):
            dmin = 1000000000
            quad_index = None
            for j, quad in enumerate(self.all_quads):
                d = distance(quad.center, sun_pos)
                if d < dmin:
                    dmin = d
                    quad_index = j

            # Add sun to the closest quad
            if quad_index is not None:
                self.all_quads[quad_index].sun_indices.append(i)
                self.all_quads[quad_index].has_sun = True

        for quad in self.all_quads:
            if quad.has_sun:
                self.active_quads.append(quad)

    def active_quad_exists(self, quad_index):
        """
        Checks if an active quad exists by its index.

        Parameters
        ----------
        quad_index : int
            Index of the quad.

        Returns
        -------
        bool
            True if the active quad exists, False otherwise.
        """
        for q in self.active_quads:
            if q.id == quad_index:
                return True
        return False

    def get_active_quad_centers(self):
        """
        Returns the centers of active quads.

        Returns
        -------
        np.ndarray
            Array of centers of active quads.
        """
        centers = []
        for quad in self.active_quads:
            centers.append(quad.center)

        return np.array(centers)

    def get_active_quad_meshes(self):
        """
        Returns the meshes of active quads.

        Returns
        -------
        list[Mesh]
            List of meshes of active quads.
        """
        meshes = []
        for quad in self.active_quads:
            meshes.append(quad.mesh)

        return meshes
