import os
import time
import numpy as np
import math
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as PPool
from multiprocessing import Process


from dtcc_io import meshes
import ncollpyde as ncp

from ncollpyde import Volume
import copy


def calc_face_midpoints(volume: Volume):
    face_vertex_index_1 = volume.faces[:, 0]
    face_vertex_index_2 = volume.faces[:, 1]
    face_vertex_index_3 = volume.faces[:, 2]
    vertex1 = volume.points[face_vertex_index_1]
    vertex2 = volume.points[face_vertex_index_2]
    vertex3 = volume.points[face_vertex_index_3]
    face_mid_pts = (vertex1 + vertex2 + vertex3) / 3.0

    vector1 = vertex2 - vertex1
    vector2 = vertex3 - vertex1

    vector_cross = np.cross(vector1, vector2)
    vector_length = np.sqrt((vector_cross**2).sum(-1))[..., np.newaxis]
    face_normals = vector_cross / vector_length

    return face_mid_pts, face_normals


def calc_rays(volume: Volume, face_mid_pts: np.ndarray, sun_vec_rev: np.ndarray):
    f_count = len(volume.faces)
    ray_start = np.zeros([f_count, 3])
    ray_end = np.zeros([f_count, 3])
    tol = 0.01
    rayLength = 1000.0
    ray_start = face_mid_pts + (sun_vec_rev * tol)
    ray_end = face_mid_pts + (sun_vec_rev * rayLength)
    return ray_start, ray_end


def raytrace(volume: Volume, ray_start, ray_end, sun_index):
    # Calculate intersections
    [seg_idxs, int_sec, is_bf] = volume.intersections(ray_start, ray_end)

    return seg_idxs, sun_index


def summation(ray_start, ray_end, sun_index):
    # Calculate intersections
    seg_idxs = ray_start + ray_end

    return seg_idxs, sun_index


if __name__ == "__main__":
    os.system("clear")

    filename = "../data/models/CitySurfaceS.stl"
    mesh = meshes.load_mesh(filename)

    volume = Volume(mesh.vertices, mesh.faces)

    face_mid_pts, face_normals = calc_face_midpoints(volume)

    sun_vec_rev1 = np.array([1.0, 1.0, 1.0])
    sun_vec_rev2 = np.array([2.0, 1.0, 1.0])
    sun_vec_rev3 = np.array([3.0, 1.0, 1.0])

    volumes = []
    volumes.append(Volume(mesh.vertices, mesh.faces))
    volumes.append(Volume(mesh.vertices, mesh.faces))
    volumes.append(Volume(mesh.vertices, mesh.faces))

    sun_vecs = []
    sun_vecs.append(sun_vec_rev1)
    sun_vecs.append(sun_vec_rev2)
    sun_vecs.append(sun_vec_rev3)

    ray_start1, ray_end1 = calc_rays(volume, face_mid_pts, sun_vec_rev1)
    ray_start2, ray_end2 = calc_rays(volume, face_mid_pts, sun_vec_rev2)
    ray_start3, ray_end3 = calc_rays(volume, face_mid_pts, sun_vec_rev3)

    ray_start = []
    ray_start.append(ray_start1)
    ray_start.append(ray_start2)
    ray_start.append(ray_start3)

    ray_end = []
    ray_end.append(ray_end1)
    ray_end.append(ray_end2)
    ray_end.append(ray_end3)

    # p = Process(target=raytrace, args=(volumes, ray_start, ray_end, sun_vecs))
    # p.start()
    # p.join()

    with PPool() as pool:
        results = pool.map(raytrace, volumes, ray_start, ray_end, sun_vecs)
        for seg_idxs, sun_index in results:
            print(sun_index, len(seg_idxs))

    # with PPool() as pool:
    #    results = pool.map(summation, ray_start, ray_end, sun_vecs)
    #    for seg_idxs, sun_index in results:
    #        print(sun_index, len(seg_idxs))

    pass
