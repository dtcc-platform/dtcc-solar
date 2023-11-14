import math
import numpy as np
from dtcc_solar.skydome import SkyDome
from dtcc_solar.utils import distance, SolarParameters, concatenate_meshes

from dtcc_solar import py_embree_solar

from dtcc_solar.sundome import SunDome
from dtcc_solar.utils import SunCollection, OutputCollection, SunApprox

from dtcc_solar.sunpath import Sunpath
from dtcc_model import Mesh, PointCloud
from dtcc_model.geometry import Bounds
from dtcc_solar.logging import info, debug, warning, error
from dtcc_viewer import Window, Scene, MeshShading
from dtcc_solar.sungroups import SunGroups
from pprint import pp


class SolarEngine:
    mesh: Mesh  # Dtcc mesh
    origin: np.ndarray
    f_count: int
    v_count: int
    horizon_z: float
    sunpath_radius: float
    sun_size: float
    path_width: float
    bb: Bounds
    bbx: np.ndarray
    bby: np.ndarray
    bbz: np.ndarray
    city_mesh_faces: np.ndarray
    city_mesh_points: np.ndarray
    city_mesh_face_mid_points: np.ndarray
    skydome: SkyDome

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh
        self.origin = np.array([0, 0, 0])
        self.f_count = len(self.mesh.faces)
        self.v_count = len(self.mesh.vertices)

        self.horizon_z = 0
        self.sunpath_radius = 0
        self.sun_size = 0
        self.dome_radius = 0
        self.path_width = 0
        self._preprocess_mesh(True)

        # Creating instance of embree solar
        self.embree = py_embree_solar.PyEmbreeSolar(mesh.vertices, mesh.faces)

        info("Solar engine created")

    def _preprocess_mesh(self, move_to_center: bool):
        self._calc_bounds()
        # Center mesh based on x and y coordinates only
        center_bb = np.array([np.average(self.bbx), np.average(self.bby), 0])
        centerVec = self.origin - center_bb

        # Move the mesh to the centre of the model
        if move_to_center:
            self.mesh.vertices += centerVec

        # Update bounding box after the mesh has been moved
        self._calc_bounds()

        # Assumption: The horizon is the avrage z height in the mesh
        self.horizon_z = np.average(self.mesh.vertices[:, 2])

        # Calculating sunpath radius
        xRange = self.bb.width
        yRange = self.bb.height
        zRange = self.zmax - self.zmin
        self.sunpath_radius = math.sqrt(
            math.pow(xRange / 2, 2) + math.pow(yRange / 2, 2) + math.pow(zRange / 2, 2)
        )

        # Hard coded size proportions
        self.sun_size = self.sunpath_radius / 90.0
        self.dome_radius = self.sunpath_radius / 40
        self.tolerance = self.sunpath_radius / 1.0e7

        info(f"Tolerance for point comparions set to: {self.tolerance}")

    def _calc_bounds(self):
        self.xmin = self.mesh.vertices[:, 0].min()
        self.xmax = self.mesh.vertices[:, 0].max()
        self.ymin = self.mesh.vertices[:, 1].min()
        self.ymax = self.mesh.vertices[:, 1].max()
        self.zmin = self.mesh.vertices[:, 2].min()
        self.zmax = self.mesh.vertices[:, 2].max()

        self.bbx = np.array([self.xmin, self.xmax])
        self.bby = np.array([self.ymin, self.ymax])
        self.bbz = np.array([self.zmin, self.zmax])

        self.bb = Bounds(xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax)

    def get_skydome_ray_count(self):
        return self.embree.get_skydome_ray_count()

    def run_analysis(self, p: SolarParameters, sunp: Sunpath, outc: OutputCollection):
        if p.sun_analysis:
            if p.sun_approx == SunApprox.quad and sunp.sundome is not None:
                self._sun_quad_raycasting(sunp.sundome, sunp.sunc, outc)
            elif p.sun_approx == SunApprox.group and sunp.sungroups is not None:
                self._sun_group_raycasting(sunp.sungroups, sunp.sunc, outc)
            elif p.sun_approx == SunApprox.none:
                self._sun_raycasting(sunp.sunc, outc)

        if p.sky_analysis:
            self._sky_raycasting(sunp.sunc, outc)

    def _sun_group_raycasting(
        self, sungroups: SunGroups, sunc: SunCollection, outc: OutputCollection
    ):
        sun_vecs = sungroups.list_centers
        sun_indices = sungroups.sun_indices
        n_groups = len(sun_vecs)

        print(f" --- Anlysing {sunc.count} suns in {len(sun_vecs)} quad groups. ---")
        if self.embree.sun_raytrace_occ8(sun_vecs):
            group_angles = np.pi - self.embree.get_angle_results()
            group_occlusion = self.embree.get_occluded_results()
            angles = np.zeros((sunc.count, self.f_count))
            occlusion = np.zeros((sunc.count, self.f_count))
            # Map the results back to the individual sun positions
            for i in range(n_groups):
                g_angles = group_angles[i]
                g_occlusion = group_occlusion[i]
                for sun_index in sun_indices[i]:
                    angles[sun_index, :] = g_angles
                    occlusion[sun_index, :] = g_occlusion

            irradiance_dn = self._calc_normal_irradiance(sunc, occlusion, angles)
            outc.face_sun_angles = angles
            outc.occlusion = occlusion
            outc.irradiance_dn = irradiance_dn
        else:
            warning(f"Something went wrong in embree solar.")

    def _sun_quad_raycasting(
        self, sundome: SunDome, sunc: SunCollection, outc: OutputCollection
    ):
        sundome.match_suns_and_quads(sunc)
        active_quads = sundome.active_quads
        sun_vecs = sundome.get_active_quad_centers()

        # active_quads_meshes = sundome.get_active_quad_meshes()
        # one_mesh = concatenate_meshes(active_quads_meshes)
        # pc = PointCloud(points=sunc.positions)
        # window = Window(1200, 800)
        # scene = Scene()
        # scene.add_mesh("Mesh diffuse", one_mesh)
        # scene.add_pointcloud("pc", pc)
        # window.render(scene)

        print(f" --- Anlysing {sunc.count} suns in {len(sun_vecs)} quad groups. ---")
        if self.embree.sun_raytrace_occ8(sun_vecs):
            quad_angles = np.pi - self.embree.get_angle_results()
            quad_occlusion = self.embree.get_occluded_results()
            angles = np.zeros((sunc.count, self.f_count))
            occlusion = np.zeros((sunc.count, self.f_count))
            # Map the results back to the individual sun positions
            for i, quad in enumerate(active_quads):
                q_angles = quad_angles[i]
                q_occlusion = quad_occlusion[i]
                for sun_index in quad.sun_indices:
                    angles[sun_index, :] = q_angles
                    occlusion[sun_index, :] = q_occlusion

            irradiance_dn = self._calc_normal_irradiance(sunc, occlusion, angles)
            outc.face_sun_angles = angles
            outc.occlusion = occlusion
            outc.irradiance_dn = irradiance_dn
        else:
            warning(f"Something went wrong in embree solar.")

    def _sun_raycasting(self, sunc: SunCollection, outc: OutputCollection):
        print(f"----- Running sun raycasting -----")
        if self.embree.sun_raytrace_occ8(sunc.sun_vecs):
            angles = np.pi - self.embree.get_angle_results()
            occlusion = self.embree.get_occluded_results()
            irradiance_dn = self._calc_normal_irradiance(sunc, occlusion, angles)

            outc.face_sun_angles = angles
            outc.occlusion = occlusion
            outc.irradiance_dn = irradiance_dn

        else:
            warning(f"Something went wrong with sun analysis in embree solar.")

    def _sky_raycasting(self, sunc: SunCollection, outc: OutputCollection):
        print(f"----- Running sky raycasting -----")
        if self.embree.sky_raytrace_occ8():  # Array with nFace elements
            facehit_sky = self.embree.get_face_skyhit_results()
            visible_sky = self.embree.get_face_skyportion_results()
            outc.visible_sky = visible_sky
            outc.facehit_sky = facehit_sky
            outc.irradiance_di = self._calc_diffuse_irradiance(sunc, visible_sky)
        else:
            warning(f"Something went wrong with sky analysis in embree solar.")

    def _calc_diffuse_irradiance(self, sunc: SunCollection, visible_sky: np.ndarray):
        diffuse_irradiance = []
        for di in sunc.irradiance_di:
            diffuse_irradiance.append(di * (1.0 - visible_sky))

        return np.array(diffuse_irradiance)

    def _calc_normal_irradiance(
        self, sunc: SunCollection, occlusion: np.ndarray, angles: np.ndarray
    ):
        irradiance = []
        for i in range(sunc.count):
            # angle_fraction = 1 if the angle is pi (180 degrees).
            angle_fraction = angles[i] / np.pi
            face_in_sun = 1.0 - occlusion[i]
            irr_for_sun = sunc.irradiance_dn[i] * face_in_sun * angle_fraction
            irradiance.append(irr_for_sun)

        return np.array(irradiance)
