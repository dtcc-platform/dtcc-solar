import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from dtcc_model import Mesh, GeometryType, MultiSurface, Surface
from dtcc_model import City, Building, Terrain
from dtcc_solar.utils import subdivide_mesh, is_mesh_valid
from dtcc_solar.logging import info, debug, warning, error


class Parts:

    face_start_index: np.ndarray
    face_end_index: np.ndarray
    ids: np.ndarray
    face_count_per_part: np.ndarray
    f_count: int

    def __init__(self, meshes: List[Mesh]):
        self._setup_parts(meshes)

    def _setup_parts(self, meshes: List[Mesh]):
        face_start_indices = []
        face_end_indices = []
        face_count_per_part = []
        tot_f_count = 0
        ids = []

        for i, mesh in enumerate(meshes):
            # Store face indices for this submesh to be used for picking
            mesh_f_count = len(mesh.faces)
            face_start_indices.append(tot_f_count)
            face_end_indices.append(tot_f_count + mesh_f_count - 1)
            tot_f_count += mesh_f_count
            face_count_per_part.append(mesh_f_count)
            ids.append(i)

        self.face_count_per_part = np.array(face_count_per_part)
        self.f_count = tot_f_count
        self.face_start_indices = np.array(face_start_indices)
        self.face_end_indices = np.array(face_end_indices)
        self.ids = np.array(ids)


def generate_building_mesh(city: City, limit: int = None, subdee_length: int = None):
    mss = []
    b = city.bounds
    origin = [(b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, (b.zmin + b.zmax) / 2]
    origin = np.array(origin)
    move_vec = origin * -1

    if limit is None:
        limit = len(city.buildings)

    for i, building in enumerate(city.buildings):
        ms = get_highest_lod_building(building)
        if i < limit:
            if isinstance(ms, MultiSurface):
                mss.append(ms)
            elif isinstance(ms, Surface):
                mss.append(MultiSurface(surfaces=[ms]))

    info(f"Found {len(mss)} building(s) in city model")

    valid_meshes = []
    for i, ms in enumerate(mss):
        mesh = ms.mesh()
        if is_mesh_valid(mesh):
            if subdee_length is not None:
                mesh = subdivide_mesh(mesh, subdee_length)
            valid_meshes.append(mesh)

    if len(valid_meshes) == 0:
        info("No building meshes found in city model")
        return None, None
    else:
        parts = Parts(valid_meshes)
        mesh = concatenate_city_meshes(valid_meshes)
        mesh.vertices += move_vec
        info(f"Mesh with {len(mesh.faces)} faces was retrieved from buildings")
        return mesh, parts


def get_terrain_mesh(city: City):
    meshes = []
    b = city.bounds
    origin = [(b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, (b.zmin + b.zmax) / 2]
    origin = np.array(origin)
    move_vec = origin * -1
    terrain_list = city.children[Terrain]
    for terrain in terrain_list:
        mesh = terrain.geometry.get(GeometryType.MESH, None)
        if mesh is not None:
            meshes.append(mesh)

    if len(meshes) == 0:
        info("No terrain mesh found in city model")
        return None, None
    else:
        mesh = concatenate_city_meshes(meshes)
        mesh.vertices += move_vec
        info(f"Terrain mesh with {len(mesh.faces)} faces was found")
        return mesh


def get_highest_lod_building(building: Building):
    lods = [
        GeometryType.LOD3,
        GeometryType.LOD2,
        GeometryType.LOD1,
        GeometryType.LOD0,
    ]

    for lod in lods:
        flat_geom = building.flatten_geometry(lod)
        if flat_geom is not None:
            return flat_geom

    return None


def concatenate_city_meshes(meshes: list[Mesh]):
    v_count_tot = 0
    f_count_tot = 0
    for mesh in meshes:
        v_count_tot += len(mesh.vertices)
        f_count_tot += len(mesh.faces)

    all_vertices = np.zeros((v_count_tot, 3), dtype=float)
    all_faces = np.zeros((f_count_tot, 3), dtype=int)

    # Accumulative face and vertex count
    acc_v_count = 0
    acc_f_count = 0

    for mesh in meshes:
        v_count = len(mesh.vertices)
        f_count = len(mesh.faces)
        vertex_offset = acc_v_count
        all_vertices[acc_v_count : acc_v_count + v_count, :] = mesh.vertices
        all_faces[acc_f_count : acc_f_count + f_count, :] = mesh.faces + vertex_offset
        acc_v_count += v_count
        acc_f_count += f_count

    mesh = Mesh(vertices=all_vertices, faces=all_faces)
    return mesh
