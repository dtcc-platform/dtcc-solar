import numpy as np
from dtcc_viewer import Scene, Window
from dtcc_model import Mesh, PointCloud
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import concatenate_meshes, SolarParameters, SunApprox
from dtcc_solar.utils import OutputCollection
from typing import Any


class Viewer:
    def __init__(self):
        self.window = Window(1200, 800)
        self.scene = Scene()

    def add_mesh(self, name: str, mesh: Mesh, data: Any = None):
        self.scene.add_mesh(name=name, mesh=mesh, data=data)

    def add_pc(
        self,
        name: str,
        pc: PointCloud,
        size: float = 0.2,
        data: np.ndarray = None,
    ):
        self.scene.add_pointcloud(name, pc=pc, size=size, data=data)

    def process_data(self, outc: OutputCollection, p: SolarParameters):
        """
        Restructure the data to a dictionary to be passed to the viewer.

        Parameters
        ----------
        outc : OutputCollection
            Collection of analysis resutls.
        p : SolarParameters
            Parameter settings.
        """
        data_dict = {}
        data_mask = outc.data_mask

        if p.sun_analysis and p.sky_analysis:
            dhi_masked = outc.dhi[data_mask] / 1000.0
            dni_masked = outc.dni[data_mask] / 1000.0
            tot_masked = dni_masked + dhi_masked
            data_dict["total irradiation (kWh/m2)"] = tot_masked

        if p.sky_analysis:
            dhi_masked = outc.dhi[data_mask] / 1000.0
            svf_masked = outc.sky_view_factor[data_mask]
            data_dict["diffuse irradiation (kWh/m2)"] = dhi_masked
            data_dict["sky view factor"] = svf_masked

        if p.sun_analysis:
            dni_masked = outc.dni[data_mask] / 1000.0
            fsa_masked = outc.face_sun_angles[data_mask]
            sun_hours_masked = outc.sun_hours[data_mask]
            shadow_hours_masked = outc.shadow_hours[data_mask]
            data_dict["direct irradiation (kWh/m2)"] = dni_masked
            data_dict["sun hours [h]"] = sun_hours_masked
            data_dict["shadow hours [h]"] = shadow_hours_masked
            data_dict["average face sun angles (rad)"] = fsa_masked

        return data_dict

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
            self.add_mesh("Sunpath mesh", mesh=quads)

        day_paths = sunpath.daypath_meshes
        day_paths = concatenate_meshes(day_paths)
        self.add_mesh("Day paths", mesh=day_paths)

        all_suns_pc = sunpath.suns_pc_minute
        self.add_pc("Suns per min", all_suns_pc, 0.2 * sunpath.w)

        sun_pc = sunpath.sun_pc
        self.add_pc("Active suns", sun_pc, 0.5 * sunpath.w)

    def show(self):
        self.window.render(self.scene)
