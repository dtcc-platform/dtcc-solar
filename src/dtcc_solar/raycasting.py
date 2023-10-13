# Raycasting function using ncollpyde
import numpy as np
import time
from dtcc_solar import utils
from ncollpyde import Volume
from dtcc_solar.utils import SunQuad
from dtcc_solar.logging import info, debug, warning, error


def raytrace(volume: Volume, sun_vec_rev: np.ndarray):
    """Raycasting function using ncollpyde"""
    mesh_faces = volume.faces
    mesh_points = volume.points
    f_count = len(mesh_faces)

    ray_start = np.zeros([f_count, 3])
    ray_end = np.zeros([f_count, 3])
    tol = 0.01
    rayLength = 1000.0

    sun_vec_rev_np = np.array(sun_vec_rev)
    face_vertex_index_1 = mesh_faces[:, 0]
    face_vertex_index_2 = mesh_faces[:, 1]
    face_vertex_index_3 = mesh_faces[:, 2]
    vertex1 = mesh_points[face_vertex_index_1]
    vertex2 = mesh_points[face_vertex_index_2]
    vertex3 = mesh_points[face_vertex_index_3]
    face_mid_pt = (vertex1 + vertex2 + vertex3) / 3.0
    ray_start = face_mid_pt + (sun_vec_rev_np * tol)
    ray_end = face_mid_pt + (sun_vec_rev_np * rayLength)

    # Calculate intersections
    [seg_idxs, int_sec, is_bf] = volume.intersections(ray_start, ray_end)

    # Rearrange intersection results
    face_in_sun = np.ones(f_count, dtype=bool)
    for ray_index in seg_idxs:
        face_in_sun[ray_index] = False

    # Return number of intersections
    n_int = len(seg_idxs)

    return face_in_sun, n_int


def raytrace_skydome(volume: Volume, ray_targets, ray_areas):
    city_mesh_faces = volume.faces
    city_mesh_points = volume.points

    tol = 0.01
    ray_scale_factor = 1000
    f_count = len(city_mesh_faces)
    ray_count = len(ray_targets)

    faceVertexIndex1 = city_mesh_faces[:, 0]
    faceVertexIndex2 = city_mesh_faces[:, 1]
    faceVertexIndex3 = city_mesh_faces[:, 2]

    vertex1 = city_mesh_points[faceVertexIndex1]
    vertex2 = city_mesh_points[faceVertexIndex2]
    vertex3 = city_mesh_points[faceVertexIndex3]

    vector1 = vertex2 - vertex1
    vector2 = vertex3 - vertex1

    vector_cross = np.cross(vector1, vector2)
    vector_length = np.sqrt((vector_cross**2).sum(-1))[..., np.newaxis]
    normal = vector_cross / vector_length

    face_mid_pt = (vertex1 + vertex2 + vertex3) / 3.0
    pt_ray_origin = face_mid_pt + (normal * tol)
    ray_targets = ray_scale_factor * ray_targets
    sky_portion = np.zeros(f_count)

    for i in range(0, f_count):
        ray_o = np.array([pt_ray_origin[i, :]])
        ray_o_repeat = np.repeat(ray_o, ray_count, axis=0)
        ray_t = ray_o_repeat + ray_targets
        [seg_idxs, intersections, is_backface] = volume.intersections(
            ray_o_repeat, ray_t
        )
        shaded_portion = np.sum(ray_areas[seg_idxs])
        sky_portion[i] = 1.0 - shaded_portion
        if (i % 100) == 0:
            info(
                f"Diffuse calculation for {i} face finished, Diffusion = {sky_portion[i]:.2}"
            )

    return sky_portion


def raytrace_sundome(volume: Volume, sun_quads: list[SunQuad]):
    faces = volume.faces
    vertices = volume.points
    fCount = len(faces)

    # Dictionary for storing the results
    res = dict.fromkeys([sun_quad.id for sun_quad in sun_quads])

    for sun_quad in sun_quads:
        if sun_quad.over_horizon:
            sun_vec_rev = np.array(sun_quad.center)

            pass
