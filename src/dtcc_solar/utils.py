import numpy as np
import math
import pandas as pd
import csv
import trimesh
import json
import fast_simplification as fs
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import defaultdict
from dtcc_model import Mesh
from csv import reader
from pandas import Timestamp, DatetimeIndex
from dtcc_solar.logging import info, debug, warning, error


class DataSource(IntEnum):
    smhi = 1
    meteo = 2
    clm = 3
    epw = 4


class MeshType(IntEnum):
    analysis = 1
    shading = 2


class SunApprox(IntEnum):
    none = 1
    group = 2
    quad = 3


class ColorBy(IntEnum):
    face_sun_angle = 1
    occlusion = 2
    irradiance_dn = 3
    irradiance_dh = 4
    irradiance_di = 5
    irradiance_tot = 6


class Mode(IntEnum):
    single_sun = 1
    multiple_sun = 2


class Analyse(Enum):
    Time = 1
    Day = 2
    Year = 3
    Times = 4


@dataclass
class Vec3:
    x: float
    y: float
    z: float


@dataclass
class SunQuad:
    face_index_a: int
    face_index_b: int
    id: int
    area: float = 0.0
    has_sun: bool = False
    over_horizon = False
    center: np.ndarray = field(default_factory=lambda: np.empty(0))
    sun_indices: List[int] = field(default_factory=list)
    mesh: Mesh = None


@dataclass
class SunCollection:
    # Number of suns
    count: int = 0
    # Sun positions
    positions: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Time stamps
    time_stamps: List[Timestamp] = field(default_factory=list)
    # Date time indices
    date_times: DatetimeIndex = None
    # Date and time of the sunposition as a string in the format: 2020-10-23T12:00:00
    datetime_strs: List[str] = field(default_factory=list)
    # Normalised sun vectors
    sun_vecs: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Direct Normal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth
    dni: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Direct Horizontal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth
    dhi: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Diffuse Horizontal Irradiance that is solar radiation diffused by athmosphere, clouds and particles
    irradiance_di: np.ndarray = field(default_factory=lambda: np.empty(0))
    # List zenith values
    zeniths: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass
class OutputCollection:
    # Date and time of the sunposition as a string in the format: 2020-10-23T12:00:00
    datetime_strs: List[str] = field(default_factory=list)
    # TimeStamp object with the same date and time
    time_stamps: List[Timestamp] = field(default_factory=list)
    # Angle between face normal an sun vector
    face_sun_angles: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Value between 0-1 for each face to determine how much of the face is occluded by other faces for given sun positions
    occlusion: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Direct Normal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth
    dni: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Direct Horizontal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth
    dhi: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Diffuse Horizontal Irradiance that is solar radiation diffused by athmosphere, clouds and particles
    irradiance_di: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Percentage of the sky dome that is visible from the face
    visible_sky: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Results for face and each sky raytracing intersection
    facehit_sky: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Number of hours of direct sun per face
    sun_hours: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Number of hours of shaded sun per face
    shadow_hours: np.ndarray = field(default_factory=lambda: np.empty(0))

    def process_results(self, face_mask: np.ndarray = None):
        fsa_masked = self.face_sun_angles[face_mask]
        dni_masked = self.dni[face_mask]
        dhi_masked = self.dhi[face_mask]
        tot_masked = dni_masked + dhi_masked
        occ_masked = self.occlusion[face_mask]
        inv_occ_masked = 1.0 - self.occlusion[face_mask]
        sun_hours_masked = self.sun_hours[face_mask]
        shadow_hours_masked = self.shadow_hours[face_mask]

        face_mask_inv = np.invert(face_mask)
        count = np.array(face_mask_inv, dtype=int).sum()

        self.data_dict_1 = {
            "total irradiance (W/m2)": tot_masked,
            "direct normal irradiance (W/m2)": dni_masked,
            "diffuse irradiance (W/m2)": dhi_masked,
            "face sun angles (rad)": fsa_masked,
            "occlusion (0-1)": occ_masked,
            "inverse occlusion (0-1)": inv_occ_masked,
            "sun hours [h]": sun_hours_masked,
            "shadow hours [h]": shadow_hours_masked,
        }

        self.data_dict_2 = None
        if face_mask is not None:
            self.data_dict_2 = {"No data": np.zeros(count)}


@dataclass
class SolarParameters:
    file_name: str = "Undefined model input file"
    weather_file: str = "Undefined weather data file"
    latitude: float = 51.5
    longitude: float = -0.12
    display: bool = True
    data_source: DataSource = DataSource.meteo
    export: bool = False
    start_date: str = "2019-06-03 07:00:00"
    end_date: str = "2019-06-03 21:00:00"
    sun_approx: SunApprox = SunApprox.group
    sun_analysis: bool = True
    sky_analysis: bool = False
    suns_per_group: int = 8
    sundome_div: tuple[int, int] = (150, 20)


def dict_2_np_array(sun_pos_dict):
    arr = []
    for key in sun_pos_dict:
        for pos in sun_pos_dict[key]:
            arr.append([pos.x, pos.y, pos.z])

    arr = np.array(arr)
    return arr


def calc_face_mid_points(mesh: Mesh):
    faceVertexIndex1 = mesh.faces[:, 0]
    faceVertexIndex2 = mesh.faces[:, 1]
    faceVertexIndex3 = mesh.faces[:, 2]

    vertex1 = mesh.vertices[faceVertexIndex1]
    vertex2 = mesh.vertices[faceVertexIndex2]
    vertex3 = mesh.vertices[faceVertexIndex3]

    face_mid_points = (vertex1 + vertex2 + vertex3) / 3.0

    return face_mid_points


def calc_face_incircle(mesh: Mesh):
    center = []
    radius = []
    for i, face in enumerate(mesh.faces):
        fv1 = face[0]
        fv2 = face[1]
        fv3 = face[2]

        A = mesh.vertices[fv1]
        B = mesh.vertices[fv2]
        C = mesh.vertices[fv3]

        vec_c = B - A
        vec_b = C - A
        vec_a = C - B

        c = np.linalg.norm(vec_c)
        b = np.linalg.norm(vec_b)
        a = np.linalg.norm(vec_a)

        s = (c + b + a) / 2.0
        f_area = 0.5 * np.linalg.norm(np.cross(vec_c, vec_b))

        c_x = (a * A[0] + b * B[0] + c * C[0]) / (2.0 * s)
        c_y = (a * A[1] + b * B[1] + c * C[1]) / (2.0 * s)
        c_z = (A[2] + B[2] + C[2]) / 3.0

        center.append([c_x, c_y, c_z])
        radius.append(f_area / s)

    return np.array(center), np.array(radius)


def create_list_of_vec3(x_list, y_list, z_list) -> List[Vec3]:
    vec_list = []
    for i in range(0, len(x_list)):
        vec = Vec3(x=x_list[i], y=y_list[i], z=z_list[i])
        vec_list.append(vec)
    return vec_list


def get_sun_vecs_from_sun_pos(sunPosList, origin):
    sunVecs = []
    for i in range(0, len(sunPosList)):
        sunPos = sunPosList[i]
        sunVec = origin - sunPos
        sunVecNorm = unitize(sunVec)
        sunVecs.append(sunVecNorm)
    return sunVecs


def unitize(vec: np.ndarray):
    length = calc_vector_length(vec)
    vecNorm = vec / length
    return vecNorm


def calc_vector_length(vec: np.ndarray):
    length = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    return length


def distance(v1, v2):
    d = math.sqrt(
        math.pow((v1[0] - v2[0]), 2)
        + math.pow((v1[1] - v2[1]), 2)
        + math.pow((v1[2] - v2[2]), 2)
    )
    return d


def concatenate_meshes(meshes: list[Mesh]):
    all_vertices = np.array([[0, 0, 0]])
    all_faces = np.array([[0, 0, 0]])
    for mesh in meshes:
        faces_offset = len(all_vertices) - 1
        all_vertices = np.vstack([all_vertices, mesh.vertices])
        faces_to_add = mesh.faces + faces_offset
        all_faces = np.vstack([all_faces, faces_to_add])

    # Remove the [0,0,0] row that was added to enable concatenate.
    all_vertices = np.delete(all_vertices, obj=0, axis=0)
    all_faces = np.delete(all_faces, obj=0, axis=0)

    mesh = Mesh(vertices=all_vertices, faces=all_faces)

    return mesh


def subdivide_mesh(mesh: Mesh, max_edge_length: float) -> Mesh:

    try:
        vs, fs = trimesh.remesh.subdivide_to_size(
            mesh.vertices, mesh.faces, max_edge=max_edge_length, max_iter=6
        )
        subdee_mesh = Mesh(vertices=vs, faces=fs)
    except:
        warning(f"Could not subdivide mesh with length {max_edge_length}.")
        subdee_mesh = mesh

    return subdee_mesh


def split_mesh_with_domain(mesh: Mesh, xdom: list, ydom: list):
    pts = calc_face_mid_points(mesh)
    x_min, y_min = pts[:, 0:2].min(axis=0)
    x_range, y_range = pts[:, 0:2].ptp(axis=0)

    if len(xdom) == len(ydom) == 2 and xdom[0] < xdom[1] and ydom[0] < ydom[1]:
        # Normalize the x,y coordinates
        xn = (pts[:, 0] - x_min) / x_range
        yn = (pts[:, 1] - y_min) / y_range
        face_mask = (xn > xdom[0]) & (xn < xdom[1]) & (yn > ydom[0]) & (yn < ydom[1])
    else:
        return None, None

    return split_mesh_by_face_mask(mesh, face_mask)


def split_mesh_by_vertical_faces(mesh: Mesh) -> Mesh:

    face_verts = mesh.vertices[mesh.faces.flatten()]
    c1 = face_verts[:-1]
    c2 = face_verts[1:]
    mask = np.ones(len(c1), dtype=bool)
    mask[2::3] = False  # [True, True, False, True, True, False, ...]
    cross_vecs = (c2 - c1)[mask]  # (v2 - v1), (v3 - v2)
    cross_p = np.cross(cross_vecs[::2], cross_vecs[1::2])  # (v2 - v1) x (v3 - v2)
    cross_p = cross_p / np.linalg.norm(cross_p, axis=1)[:, np.newaxis]  # normalize

    # Check if the cross product is pointing upwards
    mask = cross_p[:, 2] > 0.01
    mesh_1, mesh_2 = split_mesh_by_face_mask(mesh, mask)
    return mesh_2, mesh_1


def split_mesh_by_face_mask(mesh: Mesh, mask: list[bool]) -> Mesh:

    if len(mask) != len(mesh.faces):
        print("Invalid mask length")
        return None, None

    mask_inv = np.invert(mask)

    mesh_tri = trimesh.Trimesh(mesh.vertices, mesh.faces)
    mesh_tri.update_faces(mask)
    mesh_tri.remove_unreferenced_vertices()
    mesh_in = Mesh(vertices=mesh_tri.vertices, faces=mesh_tri.faces)

    mesh_tri = trimesh.Trimesh(mesh.vertices, mesh.faces)
    mesh_tri.update_faces(mask_inv)
    mesh_tri.remove_unreferenced_vertices()
    mesh_out = Mesh(vertices=mesh_tri.vertices, faces=mesh_tri.faces)

    return mesh_in, mesh_out


def is_mesh_valid(mesh: Mesh) -> bool:
    if mesh is None:
        return False

    if np.any(calc_face_areas(mesh) < 0.01):
        return False

    if len(find_dup_faces(mesh)) > 0:
        return False

    return True


def check_mesh(mesh: Mesh):
    if mesh is None:
        warning("Mesh is None")
        return

    small_faces = np.where(calc_face_areas(mesh) < 0.01)[0]
    if len(small_faces) > 0:
        warning(f"Mesh has {len(small_faces)} number of faces with area < 0.01")

    face_dups = find_dup_faces(mesh)
    if len(face_dups) > 0:
        warning(f"Mesh has {len(face_dups)} number of duplicate faces")

    vertex_dups = find_dup_vertices(mesh)
    if len(vertex_dups) > 0:
        warning(f"Mesh has {len(vertex_dups)} number of duplicate vertices")


def reduce_mesh(mesh: Mesh, reduction_rate: float = None):
    (vertices, faces) = fs.simplify(mesh.vertices, mesh.faces, reduction_rate)
    mesh = Mesh(vertices=vertices, faces=faces)
    return mesh


def find_dup_faces(mesh: Mesh) -> List[Tuple[int, int, int]]:
    """Find duplicate faces in the mesh."""
    face_set = set()
    duplicate_faces = []

    for face in mesh.faces:
        sorted_face = tuple(sorted(face))
        if sorted_face in face_set:
            duplicate_faces.append(face)
        else:
            face_set.add(sorted_face)

    return duplicate_faces


def find_dup_vertices(mesh: Mesh) -> List[int]:
    """Find duplicate vertices in the mesh."""
    vertex_dict = defaultdict(list)
    duplicate_vertices = []

    for index, vertex in enumerate(mesh.vertices):
        vertex_tuple = tuple(vertex)
        vertex_dict[vertex_tuple].append(index)

    for indices in vertex_dict.values():
        if len(indices) > 1:
            duplicate_vertices.extend(indices[1:])

    return duplicate_vertices


def calc_face_areas(mesh: Mesh) -> np.ndarray:
    """Calculate the area of each face in the mesh."""
    areas = np.zeros(mesh.num_faces)
    for i, face in enumerate(mesh.faces):
        v0, v1, v2 = (
            mesh.vertices[face[0]],
            mesh.vertices[face[1]],
            mesh.vertices[face[2]],
        )
        u = v1 - v0
        v = v2 - v0
        cross_product = np.cross(u, v)
        area = 0.5 * np.linalg.norm(cross_product)
        areas[i] = area
    return areas


def export_results(solpos):
    with open("sunpath.txt", "w") as f:
        for item in solpos["zenith"].values:
            f.write(str(item[0]) + "\n")


def print_list(listToPrint, path):
    counter = 0
    with open(path, "w") as f:
        for row in listToPrint:
            f.write(str(row) + "\n")
            counter += 1

    print("Export completed")


def print_results(shouldPrint, faceRayFaces):
    counter = 0
    if shouldPrint:
        with open("faceRayFace.txt", "w") as f:
            for key in faceRayFaces:
                f.write("Face index:" + str(key) + " " + str(faceRayFaces[key]) + "\n")
                counter += 1

    print(counter)


def write_to_csv(filename: str, data):
    print("Write to CSV")
    print(type(data))

    data_list = data.to_list()

    print(type(data_list))

    with open(filename, "w", newline="") as csvfile:
        filewriter = csv.writer(csvfile)
        for d in data:
            filewriter.writerow(str(d))


def read_sunpath_diagram_from_csv_file(filename):
    pts = []
    with open(filename, "r") as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                pt = np.array([float(row[0]), float(row[1]), float(row[2])])
                pts.append(pt)

    return pts


def match_sunpath_scale(loop_pts, radius: float):
    # Calculate the correct scale factor for the imported sunpath diagram
    pt = loop_pts[0][0]
    current_raduis = math.sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z)
    sf = radius / current_raduis

    for hour in loop_pts:
        print(len(loop_pts[hour]))
        for i in range(len(loop_pts[hour])):
            pt = loop_pts[hour][i]
            pt.x = sf * pt.x
            pt.y = sf * pt.y
            pt.z = sf * pt.z
            loop_pts[hour][i] = pt

    return loop_pts


def calc_face_mid_points(mesh):
    faceVertexIndex1 = mesh.faces[:, 0]
    faceVertexIndex2 = mesh.faces[:, 1]
    faceVertexIndex3 = mesh.faces[:, 2]
    vertex1 = np.array(mesh.vertices[faceVertexIndex1])
    vertex2 = np.array(mesh.vertices[faceVertexIndex2])
    vertex3 = np.array(mesh.vertices[faceVertexIndex3])
    face_mid_points = (vertex1 + vertex2 + vertex3) / 3.0
    return face_mid_points
