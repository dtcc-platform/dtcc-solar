import numpy as np
import math
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import List, Dict
from dtcc_model import Mesh
from dtcc_io import save_mesh


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
    sky_raycasting_some = 4  # Only for debugging.


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
class Vec3:
    x: float
    y: float
    z: float


@dataclass
class Sun:
    datetime_str: str  # Date and time of the sunposition as a string in the format: 2020-10-23T12:00:00
    datetime_ts: pd.Timestamp  # TimeStamp object with the same date and time
    index: int
    irradiance_dn: float = 0.0  # Direct Normal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth
    irradiance_dh: float = 0.0  # Direct Horizontal Irradiance from the sun beam recalculated in the normal direction in relation to the sun-earth
    irradiance_di: float = 0.0  # Diffuse Horizontal Irradiance that is solar radiation diffused by athmosphere, clouds and particles
    over_horizon: bool = (
        False  # True if the possition of over the horizon, otherwise false.
    )
    zenith: float = 0.0  # Angle between earth surface normal and the reversed solar vector (both pointing away for the earth surface)
    position: Vec3 = Vec3(
        0, 0, 0
    )  # Position of the  sun in cartesian coordinates based on the size of the model
    sun_vec: Vec3 = Vec3(0, 0, 0)  # Normalised solar vector for calculations


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
class Parameters:
    file_name: str = "Undefined model input file"
    weather_file: str = "Undefined weather data file"
    a_type: AnalysisType = AnalysisType.sky_raycasting
    latitude: float = -0.12
    longitude: float = 51.5
    prepare_display: bool = False
    display: bool = False
    data_source: DataSource = DataSource.clm
    color_by: ColorBy = ColorBy.face_sun_angle_shadows
    export: bool = False
    start_date: str = "2019-06-03 07:00:00"
    end_date: str = "2019-06-03 21:00:00"


def vec_2_ndarray(vec: Vec3):
    return np.array([vec.x, vec.y, vec.z])


def create_list_of_vectors(x_list, y_list, z_list) -> List[Vec3]:
    vector_list = []
    for i in range(0, len(x_list)):
        vec = Vec3(x=x_list[i], y=y_list[i], z=z_list[i])
        vector_list.append(vec)
    return vector_list


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
    vCross[1] = a[0] * b[2] - a[2] * b[0]
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
