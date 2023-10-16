import numpy as np
import math
import pandas as pd
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import List, Dict
from dtcc_model import Mesh
from dtcc_io import save_mesh
from csv import reader


class DataSource(IntEnum):
    smhi = 1
    meteo = 2
    clm = 3
    epw = 4


class ColorBy(IntEnum):
    face_sun_angle = 1
    face_sun_angle_shadows = 2
    face_shadows = 3
    face_irradiance_dn = 4
    face_irradiance_dh = 5
    face_irradiance_di = 6
    face_irradiance_tot = 7


class Mode(IntEnum):
    single_sun = 1
    multiple_sun = 2


class AnalysisType(IntEnum):
    sun_raycasting = 1
    sky_raycasting = 2
    com_raycasting = 3
    sun_precasting = 4
    com_precasting = 5


class AnalysisTypeDev(IntEnum):
    vertex_raycasting = 1
    subdee_shading = 2
    diffuse_dome = 3
    diffuse_some = 4
    diffuse_all = 5


class Analyse(Enum):
    Time = 1
    Day = 2
    Year = 3
    Times = 4


class Shading(Enum):
    Sun = 1
    Boarder = 2
    Shade = 3


@dataclass
class SunQuad:
    face_index_a: int
    face_index_b: int
    id: int
    area: float = 0.0
    has_sun: bool = False
    over_horizon = False
    center: np.ndarray = field(default_factory=lambda: np.empty(0))
    sun_indices: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass
class Vec3:
    x: float
    y: float
    z: float


@dataclass
class Sun:
    # Date and time of the sunposition as a string in the format: 2020-10-23T12:00:00
    datetime_str: str
    # TimeStamp object with the same date and time
    datetime_ts: pd.Timestamp
    index: int
    # Direct Normal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth
    irradiance_dn: float = 0.0
    # Direct Horizontal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth
    irradiance_dh: float = 0.0
    # Diffuse Horizontal Irradiance that is solar radiation diffused by athmosphere, clouds and particles
    irradiance_di: float = 0.0
    # True if the possition of over the horizon, otherwise false.
    over_horizon: bool = False
    # Angle between earth surface normal and the reversed solar vector (both pointing away for the earth surface)
    zenith: float = 0.0
    # Position of the  sun in cartesian coordinates based on the size of the model
    position: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    # Normalised solar vector for calculations
    sun_vec: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))


@dataclass
class Output:
    datetime_str: str  # Date and time of the sunposition as a string in the format: 2020-10-23T12:00:00
    datetime_ts: pd.Timestamp  # TimeStamp object with the same date and time
    index: int
    face_sun_angles: List[float]
    face_in_sun: List[bool]
    face_in_sky: List[bool]
    face_irradiance_dn: List[float]
    face_irradiance_dh: List[float]
    face_irradiance_di: List[float]
    face_irradiance_tot: List[float]


@dataclass
class OutputAcum:
    start_datetime_str: str  # Date and time of the sunposition as a string in the format: 2020-10-23T12:00:00
    start_datetime_ts: pd.Timestamp
    end_datetime_str: str  # Date and time of the sunposition as a string in the format: 2020-10-23T12:00:00
    end_datetime_ts: pd.Timestamp
    face_sun_angles: List[float]
    face_in_sun: List[bool]
    face_in_sky: List[bool]
    face_irradiance_dn: List[float]
    face_irradiance_dh: List[float]
    face_irradiance_di: List[float]
    face_irradiance_tot: List[float]


@dataclass
class SolarParameters:
    file_name: str = "Undefined model input file"
    weather_file: str = "Undefined weather data file"
    a_type: AnalysisType = AnalysisType.sky_raycasting
    latitude: float = 51.5
    longitude: float = -0.12
    display: bool = True
    data_source: DataSource = DataSource.clm
    color_by: ColorBy = ColorBy.face_sun_angle_shadows
    export: bool = False
    start_date: str = "2019-06-03 07:00:00"
    end_date: str = "2019-06-03 21:00:00"


def vec_2_ndarray(vec: Vec3):
    return np.array([vec.x, vec.y, vec.z])


def create_list_of_vec3(x_list, y_list, z_list) -> List[Vec3]:
    vec_list = []
    for i in range(0, len(x_list)):
        vec = Vec3(x=x_list[i], y=y_list[i], z=z_list[i])
        vec_list.append(vec)
    return vec_list


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


"""
def find_distributed_face_mid_points(nx, ny, mesh:trimesh):

    bb = trimesh.bounds.corners(mesh.bounding_box.bounds)
    bbx = np.array([np.min(bb[:,0]), np.max(bb[:,0])])
    bby = np.array([np.min(bb[:,1]), np.max(bb[:,1])])
    bbz = np.array([np.min(bb[:,2]), np.max(bb[:,2])])

    dx = (bbx[1] - bbx[0])/(nx-1)
    dy = (bby[1] - bby[0])/(ny-1)

    x = bbx[0]
    y = bby[0]
    z = np.average([bbz])

    face_mid_points = calc_face_mid_points(mesh)
    face_mid_points[:,2] = z 
    face_indices = []

    for i in range(0,nx):
        y = bby[0]
        for j in range(0,ny):
            point = np.array([x,y,z])
            index = get_index_of_closest_point(point, face_mid_points)
            face_indices.append(index)
            y += dy
        x += dx

    return face_indices        

def calc_face_mid_points(mesh:trimesh):

    faceVertexIndex1 = mesh.faces[:,0]
    faceVertexIndex2 = mesh.faces[:,1]
    faceVertexIndex3 = mesh.faces[:,2] 

    vertex1 = mesh.vertices[faceVertexIndex1]
    vertex2 = mesh.vertices[faceVertexIndex2]
    vertex3 = mesh.vertices[faceVertexIndex3]

    face_mid_points = (vertex1 + vertex2 + vertex3)/3.0

    return face_mid_points
"""


def get_sun_vecs_from_sun_pos(sunPosList, origin):
    sunVecs = []
    for i in range(0, len(sunPosList)):
        sunPos = sunPosList[i]
        sunVec = origin - sunPos
        sunVecNorm = normalise_vector(sunVec)
        sunVecs.append(sunVecNorm)
    return sunVecs


def normalise_vector(vec):
    length = calc_vector_length(vec)
    vecNorm = np.zeros(3)
    vecNorm = vec / length
    return vecNorm


def normalise_vector3(vec: Vec3):
    length = calc_vector_length3(vec)
    vecNorm = Vec3(x=(vec.x / length), y=(vec.y / length), z=(vec.z / length))
    return vecNorm


def reverse_vector(vec):
    vecRev = -1.0 * vec
    return vecRev


def calc_vector_length(vec):
    length = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    return length


def calc_vector_length3(vec: Vec3):
    length = math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)
    return length


def scale_vector3(vec: Vec3, sf: float):
    scaled_vec = Vec3(x=sf * vec.x, y=sf * vec.y, z=sf * vec.z)
    return scaled_vec


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
    normal = normalise_vector(v3)
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
                pt = Vec3(x=float(row[0]), y=float(row[1]), z=float(row[2]))
                pts.append(pt)

    return pts


def match_sunpath_scale(
    loop_pts: Dict[int, List[Vec3]], radius: float
) -> Dict[int, List[Vec3]]:
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
