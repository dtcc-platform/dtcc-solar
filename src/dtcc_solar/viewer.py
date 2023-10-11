import numpy as np
from dtcc_viewer import Scene, Window, MeshShading
from dtcc_model import Mesh, PointCloud
from dtcc_solar.sunpath_vis import SunpathMesh
from dtcc_solar.utils import Sun
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import concatenate_meshes
from dtcc_solar.skycylinder import SkyCylinder


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

    def create_sunpath_diagram(
        self,
        suns: list[Sun],
        solar_engine: SolarEngine,
        sunpath: Sunpath,
        skycylinder: SkyCylinder,
    ):
        # Create sunpath so that the solar postion are given a context in the 3D visualisation
        self.sunpath_mesh = SunpathMesh(solar_engine.sunpath_radius)
        self.sunpath_mesh.create_sunpath_diagram(suns, sunpath, solar_engine)

        # Get analemmas, day paths, and pc for sun positions
        analemmas = self.sunpath_mesh.analemmas_meshes
        day_paths = self.sunpath_mesh.daypath_meshes
        analemmas_pc = self.sunpath_mesh.analemmas_pc
        all_suns_pc = self.sunpath_mesh.all_suns_pc
        sun_pc = self.sunpath_mesh.sun_pc
        sky_mesh = skycylinder.mesh

        analemmas = concatenate_meshes(analemmas)
        day_paths = concatenate_meshes(day_paths)

        self.add_mesh("Sunpath mesh", mesh=sky_mesh, shading=MeshShading.wireframe)
        self.add_mesh("Analemmas", mesh=analemmas, shading=MeshShading.ambient)
        self.add_mesh("Day paths", mesh=day_paths, shading=MeshShading.ambient)
        self.add_pc("Suns per min", all_suns_pc, 0.2 * solar_engine.path_width)
        self.add_pc("Analemmas suns", analemmas_pc, 0.5 * solar_engine.path_width)
        self.add_pc("Active suns", sun_pc, 5 * solar_engine.path_width)

    def show(self):
        self.window.render(self.scene)
