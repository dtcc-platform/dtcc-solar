import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from dtcc_core.model import Mesh, GeometryType, MultiSurface, Surface
from dtcc_core.model import City, Building, Terrain
from dtcc_core import builder
from dtcc_solar.utils import subdivide_mesh, is_mesh_valid, SolarParameters
from dtcc_solar.utils import OutputCollection
from dtcc_solar.logging import info, debug, warning, error
from enum import Enum, IntEnum


def get_building_meshes(city: City):
    meshes = []
    for b in city.buildings:
        geom = get_highest_lod_building(b)
        if geom is not None:
            mesh = geom.mesh(weld=True)
            meshes.append(mesh)

    return meshes


def get_terrain(city: City):
    if city.terrain is None:
        info("No terrain found in city model")
        return None

    return city.terrain.mesh


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


def get_highest_lod(building: Building):
    lods = [
        GeometryType.LOD3,
        GeometryType.LOD2,
        GeometryType.LOD1,
        GeometryType.LOD0,
    ]

    for lod in lods:
        flat_geom = building.flatten_geometry(lod)
        if flat_geom is not None:
            return lod

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


def export_results_to_json(
    face_count,
    p: SolarParameters,
    outputc: OutputCollection,
    filename,
):
    """Export a mesh and its associated data to a JSON file."""

    info(f"Exporting mesh to {filename}")

    svf = outputc.sky_view_factor[outputc.data_mask]
    sun_hours = outputc.sun_hours[outputc.data_mask]

    # Ensure the length of data lists matches the number of faces
    assert len(svf) == face_count
    assert len(sun_hours) == face_count

    parameters = {
        "file_name": p.file_name,
        "sun_analysis": p.sun_analysis,
        "sky_analysis": p.sky_analysis,
        "start_date": p.start_date,
        "end_date": p.end_date,
        "data_source": p.data_source,
        "latitude": p.latitude,
        "longitude": p.longitude,
        "data_source": p.data_source,
        "weather_data": p.weather_file,
    }

    # Create the structure to hold the mesh data
    results_data = {
        "SkyViewFactor": svf.tolist(),
        "SunHours": sun_hours.tolist(),
        "Parameters": parameters,
    }

    # Write the data to a JSON file
    with open(filename, "w") as json_file:
        json.dump(results_data, json_file, indent=4)

    info(f"Results exported successfully")
