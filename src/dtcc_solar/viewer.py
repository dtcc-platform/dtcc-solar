import numpy as np
from dtcc_viewer import Scene, Window, MeshShading
from dtcc_model import Mesh, PointCloud


class Viewer:
    def __init__(self):
        self.window = Window(1200, 800)
        self.scene = Scene()

    def add_mesh(
        self,
        name: str,
        mesh: Mesh,
        colors: np.ndarray = None,
        data: np.ndarray = None,
        shading: MeshShading = MeshShading.diffuse,
    ):
        self.scene.add_mesh(
            name=name, mesh=mesh, colors=colors, data=data, shading=shading
        )

    def add_pc(
        self,
        name: str,
        pc: PointCloud,
        size: float = 0.2,
        colors: np.ndarray = None,
        data: np.ndarray = None,
    ):
        self.scene.add_pointcloud(name, pc=pc, size=size, colors=colors, data=data)

    def show(self):
        self.window.render(self.scene)
