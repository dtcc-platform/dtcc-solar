import numpy as np
from dtcc_viewer import Scene, Window
from dtcc_model import Mesh, PointCloud
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import concatenate_meshes, SolarParameters
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
        data_mask = outc.data_mask  # Mask for faces with data

        print("Data mask:", data_mask)
        print("Data mask shape:", data_mask.shape)
        print("face sun angles shape", outc.face_sun_angles.shape)

        if p.sky_analysis:
            svf_masked = outc.sky_view_factor[data_mask]
            data_dict["sky view factor"] = svf_masked

        if p.sun_analysis:
            fsa_masked = outc.face_sun_angles[data_mask]
            sun_hours_masked = outc.sun_hours[data_mask]
            shadow_hours_masked = outc.shadow_hours[data_mask]
            data_dict["sun hours [h]"] = sun_hours_masked
            data_dict["shadow hours [h]"] = shadow_hours_masked
            data_dict["average face sun angles (rad)"] = fsa_masked

        return data_dict

    def build_sunpath_diagram(self, sunpath: Sunpath, p: SolarParameters):
        day_paths = sunpath.daypath_meshes
        day_paths = concatenate_meshes(day_paths)
        all_suns_pc = sunpath.suns_pc_minute
        sun_pc = sunpath.sun_pc
        self.scene.add_mesh(name="day paths", mesh=day_paths)
        self.scene.add_pointcloud("sun per minute", all_suns_pc, 0.2 * sunpath.w)
        self.scene.add_pointcloud("active suns", sun_pc, 0.5 * sunpath.w)

    def show(self):
        self.window.render(self.scene)
