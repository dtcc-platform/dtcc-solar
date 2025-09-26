import os
import sys
import numpy as np
from dtcc_viewer import Scene, Window
from dtcc_core.model import Mesh, PointCloud
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import concatenate_meshes, SolarParameters, create_ls_circle
from dtcc_solar.utils import OutputCollection, AnalysisType, split_mesh_by_face_mask
from dtcc_solar.logging import info, debug, warning, error
from dtcc_solar.skydome import Skydome
from typing import Any


class Viewer:

    has_display: bool

    def __init__(
        self,
        output: OutputCollection,
        skydome: Skydome,
        sunpath: Sunpath,
        p: SolarParameters,
    ):

        self.has_display = has_display()

        if self.has_display and p.display:
            self.window = Window(1200, 800)
            self.scene = Scene()
            self.prepare_scene(output, skydome, sunpath, p)
            self.show()
        else:
            self.has_display = False
            info("No display detected - running headless.")

        return

    def prepare_scene(
        self,
        output: OutputCollection,
        skydome: Skydome,
        sunpath: Sunpath,
        p: SolarParameters,
    ):
        """Prepare the scene for viewing."""

        sunpath.create_sunpath_geometry()
        data_dict = {}
        mask = output.data_mask
        r = sunpath.r
        a_mesh, s_mesh = split_mesh_by_face_mask(output.mesh, mask)

        if p.analysis_type == AnalysisType.TWO_PHASE:
            data_dict["total_irradiance (kW/m²)"] = output.total_irradiance[mask]

        if p.analysis_type == AnalysisType.THREE_PHASE:
            data_dict["total_irradiance (kW/m²)"] = output.total_irradiance[mask]
            data_dict["sky_irradiance (kW/m²)"] = output.sky_irradiance[mask]
            data_dict["sun_irradiance (kW/m²)"] = output.sun_irradiance[mask]

        self.scene.add_mesh(name="Analysis mesh", mesh=a_mesh, data=data_dict)

        if output.shading_mesh is not None:
            self.scene.add_mesh(name="Shading mesh", mesh=s_mesh)

        if p.analysis_type == AnalysisType.TWO_PHASE:
            self.build_skydome_2_phase(output, skydome, r)
        if p.analysis_type == AnalysisType.THREE_PHASE:
            self.build_skydome_3_phase(output, skydome, r)
            self.build_sunpath_diagram(sunpath, p)

    def build_skydome_2_phase(
        self, output: OutputCollection, skydome: Skydome, r: float
    ):
        """Build the skydome for the scene."""
        data_dict = {}
        mesh = skydome.mesh
        mesh.vertices *= r
        sky_vec = np.sum(output.sky_results.matrix, axis=1)
        sun_vec = np.sum(output.sun_results.matrix, axis=1)
        sky_vec = skydome.map_data_to_faces(sky_vec) * 0.001  # W to kW
        sun_vec = skydome.map_data_to_faces(sun_vec) * 0.001  # W to kW
        tot_vec = sky_vec + sun_vec
        data_dict["tot matrix"] = tot_vec
        data_dict["sky matrix"] = sky_vec
        data_dict["sun matrix"] = sun_vec
        self.scene.add_mesh(name="Skydome", mesh=skydome.mesh, data=data_dict)

    def build_skydome_3_phase(
        self, output: OutputCollection, skydome: Skydome, r: float
    ):
        """Build the skydome for the scene."""
        data_dict = {}
        mesh = skydome.mesh
        mesh.vertices *= r
        sky_vec = np.sum(output.sky_results.matrix, axis=1)
        sky_vec = skydome.map_data_to_faces(sky_vec) * 0.001  # W to kW
        data_dict["sky matrix"] = sky_vec
        self.scene.add_mesh(name="Skydome", mesh=skydome.mesh, data=data_dict)

    def build_sunpath_diagram(self, sunpath: Sunpath, p: SolarParameters):
        day_paths = sunpath.daypath_meshes
        day_paths = concatenate_meshes(day_paths)
        sun_pc = sunpath.sun_pc
        self.scene.add_mesh(name="day paths", mesh=day_paths)
        self.scene.add_pointcloud("suns", sun_pc, 0.5 * sunpath.w)

    def show(self):
        if self.has_display:
            self.window.render(self.scene)


class SkydomeViewer:
    has_display: bool

    def __init__(
        self, skydome: Skydome, patch_data: dict, sun_pos_pc: PointCloud = None
    ):
        self.has_display = has_display()

        if self.has_display:
            self.window = Window(1200, 800)
            self.scene = Scene()
            self.prepare_scene(skydome, patch_data, sun_pos_pc)
            self.show()
        else:
            self.has_display = False
            info("No display detected -> can't show skydome.")

        return

    def prepare_scene(
        self, skydome: Skydome, data_dict: dict, sun_pos_pc: PointCloud = None
    ):
        """Prepare the scene for viewing."""
        mesh = skydome.mesh
        mapped_data_dict = None
        if data_dict is not None and isinstance(data_dict, dict):
            mapped_data_dict = skydome.map_dict_data_to_faces(data_dict)

        ls_cirlce = create_ls_circle(np.array([0, 0, 0]), skydome.r * 2, 100)
        self.scene.add_mesh(name="Skydome", mesh=mesh, data=mapped_data_dict)
        self.scene.add_linestring("Skydome circle", ls_cirlce)
        if sun_pos_pc is not None:
            self.scene.add_pointcloud("Sun positions", sun_pos_pc, size=0.01)

    def show(self):
        if self.has_display:
            self.window.render(self.scene)


def has_display():
    if sys.platform.startswith("linux"):
        return "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ
    elif sys.platform == "darwin":
        # On macOS, assume display is present unless running headless (e.g. over SSH)
        return "SSH_CONNECTION" not in os.environ
    else:
        # Windows and others: assume GUI available
        return True
