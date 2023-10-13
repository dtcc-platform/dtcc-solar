import numpy as np
import trimesh
import copy

from dtcc_solar import utils
from trimesh import Trimesh


def compute_irradiance(face_in_sun, face_angles, f_count, flux):
    irradiance = np.zeros(f_count)
    for i in range(0, f_count):
        # angle_fraction = 1 if the angle is pi (180 degrees).
        angle_fraction = face_angles[i] / np.pi
        face_in_sun_int = float(face_in_sun[i])
        irradiance[i] = flux * face_in_sun_int * angle_fraction

    return irradiance


def face_sun_angle(mesh: Trimesh, sunVec):
    face_sun_angles = np.zeros(len(mesh.faces))
    mesh_faces = list(mesh.faces)
    mesh_face_normals = list(mesh.face_normals)
    for i in range(0, len(mesh_faces)):
        face_normal = mesh_face_normals[i]
        face_sun_angle = 0.0
        face_sun_angle = utils.vector_angle(sunVec, face_normal)
        face_sun_angles[i] = face_sun_angle

    return face_sun_angles
