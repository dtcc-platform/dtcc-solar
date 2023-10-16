import numpy as np
from dtcc_viewer import Scene, Window, MeshShading
from dtcc_model import Mesh, PointCloud
from dtcc_solar.utils import Sun
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import concatenate_meshes
from dtcc_solar.sundome import SunDome


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

    def build_sunpath_diagram(
        self,
        sunpath: Sunpath,
        sundome: SunDome,
    ):
        # Get analemmas, day paths, and pc for sun positions
        analemmas = sunpath.analemmas_meshes
        day_paths = sunpath.daypath_meshes
        analemmas_pc = sunpath.analemmas_pc
        all_suns_pc = sunpath.all_suns_pc
        sun_pc = sunpath.sun_pc
        sky_mesh = sundome.mesh

        analemmas = concatenate_meshes(analemmas)
        day_paths = concatenate_meshes(day_paths)

        self.add_mesh("Sunpath mesh", mesh=sky_mesh, shading=MeshShading.wireframe)
        self.add_mesh("Analemmas", mesh=analemmas, shading=MeshShading.ambient)
        self.add_mesh("Day paths", mesh=day_paths, shading=MeshShading.ambient)
        self.add_pc("Suns per min", all_suns_pc, 0.2 * sunpath.w)
        self.add_pc("Analemmas suns", analemmas_pc, 0.5 * sunpath.w)
        self.add_pc("Active suns", sun_pc, 5 * sunpath.w)

    def show(self):
        self.window.render(self.scene)
