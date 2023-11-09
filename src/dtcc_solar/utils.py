import numpy as np
import math
import pandas as pd
import csv
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import List, Dict
from dtcc_model import Mesh
from dtcc_io import save_mesh
from csv import reader
from pandas import Timestamp, DatetimeIndex


class DataSource(IntEnum):
    smhi = 1
    meteo = 2
    clm = 3
    epw = 4


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
    irradiance_dn: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Direct Horizontal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth
    irradiance_dh: np.ndarray = field(default_factory=lambda: np.empty(0))
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
    # Angle between face normal an sun vector
    occlusion: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Direct Normal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth
    irradiance_dn: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Direct Horizontal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth
    irradiance_dh: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Diffuse Horizontal Irradiance that is solar radiation diffused by athmosphere, clouds and particles
    irradiance_di: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Percentage of the sky dome that is visible from the face
    visible_sky: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Results for face and each sky raytracing intersection
    facehit_sky: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass
class SolarParameters:
    file_name: str = "Undefined model input file"
    weather_file: str = "Undefined weather data file"
    latitude: float = 51.5
    longitude: float = -0.12
    display: bool = True
    data_source: DataSource = DataSource.clm
    color_by: ColorBy = ColorBy.face_sun_angle
    export: bool = False
    start_date: str = "2019-06-03 07:00:00"
    end_date: str = "2019-06-03 21:00:00"
    use_quads: bool = False
    sun_analysis: bool = True
    sky_analysis: bool = False


def dict_2_np_array(sun_pos_dict):
    arr = []
    for key in sun_pos_dict:
        for pos in sun_pos_dict[key]:
            arr.append([pos.x, pos.y, pos.z])

    arr = np.array(arr)
    return arr


def calc_rotation_matrix(vec_from: np.ndarray, vec_to: np.ndarray):
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    a = vec_from
    b = vec_to

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    a = a / a_norm
    b = b / b_norm

    v = np.cross(a, b)

    angle = math.acos(
        ((a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])) / (a_norm * b_norm)
    )

    s = np.linalg.norm(v) * math.sin(angle)
    c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) * math.cos(angle)

    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    constant = 1.0 / (1.0 + c)

    R = I + Vx + np.dot(Vx, Vx) * constant

    return R


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


def reverse_vector(vec):
    vecRev = -1.0 * vec
    return vecRev


def calc_vector_length(vec: np.ndarray):
    length = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    return length


def vector_angle(vec1, vec2):
    lengthV1 = calc_vector_length(vec1)
    lengthV2 = calc_vector_length(vec2)
    scalarV1V2 = scalar_product(vec1, vec2)
    denominator = lengthV1 * lengthV2
    if denominator != 0:
        angle = math.acos(scalarV1V2 / (denominator))
        return angle
    return 0


def calculate_normal(v1, v2):
    v3 = cross_product(v1, v2)
    normal = unitize(v3)
    return normal


def cross_product(a, b):
    vCross = np.zeros(3)
    vCross[0] = a[1] * b[2] - a[2] * b[1]
    vCross[1] = a[2] * b[0] - a[0] * b[2]
    vCross[2] = a[0] * b[1] - a[1] * b[0]
    return vCross


def scalar_product(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


def distance(v1, v2):
    return math.sqrt(
        math.pow((v1[0] - v2[0]), 2)
        + math.pow((v1[1] - v2[1]), 2)
        + +math.pow((v1[2] - v2[2]), 2)
    )


def reverse_mask(mask):
    revMask = [not elem for elem in mask]
    return revMask


def get_index_of_closest_point(point, array_of_points):
    dmin = 10000000000
    index = -1
    for i in range(0, len(array_of_points)):
        d = distance(point, array_of_points[i])
        if d < dmin:
            dmin = d
            index = i

    return index


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


def print_dict(dictToPrint, filename):
    counter = 0
    with open(filename, "w") as f:
        for key in dictToPrint:
            f.write("Key:" + str(key) + " " + str(dictToPrint[key]) + "\n")


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
