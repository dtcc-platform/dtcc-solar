import numpy as np
import pandas as pd
import trimesh
import math
import copy

from dtcc_solar import utils
from dtcc_solar.skydome import SkyDome
from dtcc_model import Mesh
from dtcc_solar.logging import info, debug, warning, error


class SkySphere:
    div_count: int
    radius: float
    skydome: SkyDome
    mesh: Mesh

    def __init__(self, sphere_radius, div_count: int = 10):
        self.div_count = div_count
        self.radius = sphere_radius
        self.skydome = SkyDome(self.radius, self.div_count)
        self.build_mesh()

    def build_mesh(self):
        vs = self.skydome.dome_mesh.vertices
        fs = self.skydome.dome_mesh.faces
        mesh_upper = Mesh(vertices=vs, faces=fs)

        vs = copy.deepcopy(mesh_upper.vertices)
        vs[:, 2] *= -1

        rev_faces = []
        for face in fs:
            rev_face = [face[0], face[2], face[1]]
            rev_faces.append(rev_face)

        rev_faces = np.array(rev_faces)
        mesh_lower = Mesh(vertices=vs, faces=rev_faces)
        self.mesh = utils.concatenate_meshes([mesh_lower, mesh_upper])

        info("Sky dome mesh created")
