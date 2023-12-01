import numpy as np
import pprint
from dtcc_solar.logging import info, debug, warning, error
from dtcc_model import Mesh, PointCloud
from utils import MeshType, concatenate_meshes
from typing import Any


class SolarModel:
    analysis_mesh: Mesh
    shading_mesh: Mesh
    combined_mesh: Mesh  # To be sent to Embree for raytracing
    face_mask: np.ndarray  # Also sent to Embree for raytracing
    mesh_types: list[MeshType]
    amesh_face_indices: np.ndarray
    smesh_face_indices: np.ndarray

    def __init__(self, meshes: Any, types: Any) -> None:
        self._setup(meshes, types)

    def _setup(self, meshes, mtypes):
        if isinstance(meshes, list) and isinstance(mtypes, list):
            if len(meshes) == len(mtypes):
                self._setup_mesh_list(meshes, mtypes)
            else:
                warning("The number of meshes and types do not match.")
        elif isinstance(meshes, Mesh) and isinstance(mtypes, MeshType):
            if isinstance(meshes, Mesh) and isinstance(mtypes, MeshType):
                self._setup_mesh(meshes, mtypes)
            else:
                warning("Type error.")
        else:
            warning("Solar model type error in constructor.")

    def _setup_mesh(self, mesh, mtype):
        self.combined_mesh = mesh

    def _setup_mesh_list(self, meshes: list[Mesh] = None, types: list[MeshType] = None):
        if len(meshes) == len(types):
            face_mask = []
            aface_indices = []
            sface_indices = []
            face_index_1 = 0
            face_index_2 = 0
            mask = 0
            for i, mesh in enumerate(meshes):
                face_index_2 = len(mesh.faces)
                if types[i] == MeshType.analysis:
                    mask = np.ones(len(mesh.faces), dtype=bool)
                    aface_indices.append(np.arange(face_index_1, face_index_2))
                elif types[i] == MeshType.analysis:
                    mask = np.zeros(len(mesh.faces), dtype=bool)
                    sface_indices.append(np.arange(face_index_1, face_index_2))

                face_mask.extend(mask)

            # Save data for analysis and retrevial of original meshes
            self.combined_mesh = concatenate_meshes(meshes)
            self.face_mask = face_mask
            self.amesh_face_indices = aface_indices
            self.smesh_face_indices = sface_indices

        else:
            warning("The number of meshes and types do not match.")
