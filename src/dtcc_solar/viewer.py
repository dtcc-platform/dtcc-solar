import numpy as np
from dtcc_viewer import Scene, Window, MeshShading
from dtcc_model import Mesh, PointCloud
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import concatenate_meshes, SolarParameters, SunApprox
from dtcc_solar.sundome import SunDome
from dtcc_solar.sungroups import SunGroups
from typing import Any


class Viewer:
    def __init__(self):
        self.window = Window(1200, 800)
        self.scene = Scene()

    def add_mesh(
        self,
        name: str,
        mesh: Mesh,
        data: Any = None,
        shading: MeshShading = MeshShading.diffuse,
    ):
        self.scene.add_mesh(name=name, mesh=mesh, data=data, shading=shading)

    def add_pc(
        self,
        name: str,
        pc: PointCloud,
        size: float = 0.2,
        data: np.ndarray = None,
    ):
        self.scene.add_pointcloud(name, pc=pc, size=size, data=data)

    def build_sunpath_diagram(self, sunpath: Sunpath, p: SolarParameters):
        if p.sun_approx == SunApprox.group:
            group_centers_pc = sunpath.sungroups.centers_pc
            self.add_pc("Group centers", group_centers_pc, 1.5 * sunpath.w)
        elif p.sun_approx == SunApprox.quad:
            # sky_mesh = sunpath.sundome.mesh
            quads = sunpath.sundome.get_active_quad_meshes()
            centers = sunpath.sundome.get_active_quad_centers()
            centers_pc = PointCloud(points=centers)
            quads = concatenate_meshes(quads)
            self.add_pc("Quad centers", centers_pc, 1.5 * sunpath.w)
            self.add_mesh("Sunpath mesh", mesh=quads, shading=MeshShading.wireframe)

        day_paths = sunpath.daypath_meshes
        day_paths = concatenate_meshes(day_paths)
        self.add_mesh("Day paths", mesh=day_paths, shading=MeshShading.ambient)

        all_suns_pc = sunpath.all_suns_pc
        self.add_pc("Suns per min", all_suns_pc, 0.2 * sunpath.w)

        sun_pc = sunpath.sun_pc
        self.add_pc("Active suns", sun_pc, 0.5 * sunpath.w)

    def show(self):
        self.window.render(self.scene)
