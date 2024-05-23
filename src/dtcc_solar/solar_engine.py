import math
import numpy as np
from dtcc_solar.utils import SolarParameters, calc_face_mid_points, concatenate_meshes
from dtcc_solar import py_embree_solar as embree
from dtcc_solar.sundome import SunDome
from dtcc_solar.utils import SunCollection, OutputCollection, SunApprox

from dtcc_solar.sunpath import Sunpath
from dtcc_model import Mesh, PointCloud
from dtcc_model.geometry import Bounds
from dtcc_solar.logging import info, debug, warning, error
from dtcc_viewer import Window, Scene
from dtcc_solar.sungroups import SunGroups
from pprint import pp


class SolarEngine:
    mesh: Mesh  # Mesh for analysis
    shading_mesh: Mesh  # Mesh that will cast shadows but not be analysed
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
    face_areas: np.ndarray
    face_mask: np.ndarray

    def __init__(self, analysis_mesh: Mesh, shading_mesh: Mesh = None):
        self.analysis_mesh = analysis_mesh
        self.shading_mesh = shading_mesh
        (self.mesh, self.face_mask) = self._join_meshes(analysis_mesh, shading_mesh)
        self.origin = np.array([0, 0, 0])
        self.f_count = len(self.mesh.faces)
        self.v_count = len(self.mesh.vertices)
        info(f"Mesh has {self.f_count} faces and {self.v_count} vertices.")

        self.horizon_z = 0
        self.sunpath_radius = 0
        self.sun_size = 0
        self.dome_radius = 0
        self.path_width = 0
        self._preprocess_mesh(True)

        info("Solar engine created")

    def _join_meshes(self, analysis_mesh: Mesh, shading_mesh: Mesh = None):
        if shading_mesh is None:
            return analysis_mesh, np.ones(len(analysis_mesh.faces), dtype=bool)

        mesh = concatenate_meshes([analysis_mesh, shading_mesh])
        face_mask = np.ones(len(analysis_mesh.faces), dtype=bool)
        face_mask = np.append(face_mask, np.zeros(len(shading_mesh.faces), dtype=bool))
        return mesh, face_mask

    def _preprocess_mesh(self, move_to_center: bool):
        self._calc_bounds()
        # Center mesh based on x and y coordinates only
        center_bb = np.array(
            [np.average(self.bbx), np.average(self.bby), np.average(self.bbz)]
        )
        centerVec = self.origin - center_bb

        # Move the mesh to the centre of the model
        if move_to_center:
            self.mesh.vertices += centerVec

        # Update bounding box after the mesh has been moved
        self._calc_bounds()

        # Assumption: The horizon is the avrage z height in the mesh
        self.horizon_z = np.average(self.mesh.vertices[:, 2])

        # Calculating sunpath radius
        dx = self.bb.width
        dy = self.bb.height
        dz = self.zmax - self.zmin
        self.sunpath_radius = math.sqrt(
            math.pow(dx / 2, 2) + math.pow(dy / 2, 2) + math.pow(dz / 2, 2)
        )

        # Hard coded size proportions
        self.sun_size = self.sunpath_radius / 90.0
        self.dome_radius = self.sunpath_radius / 40
        self.tolerance = self.sunpath_radius / 1.0e7

        self._calc_face_areas()
        if move_to_center:
            info(f"Mesh has been moved to origin.")

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

    def _calc_face_areas(self):
        areas = []
        for face in self.mesh.faces:
            v1 = self.mesh.vertices[face[0]]
            v2 = self.mesh.vertices[face[1]]
            v3 = self.mesh.vertices[face[2]]
            v1v2 = v2 - v1
            v1v3 = v3 - v1
            area = 0.5 * np.linalg.norm(np.cross(v1v2, v1v3))
            areas.append(area)

        self.face_areas = np.array(areas)

    def get_skydome_ray_count(self):
        return self.embree.get_skydome_ray_count()

    def run_analysis(self, p: SolarParameters, sunp: Sunpath, outc: OutputCollection):
        # Creating instance of embree solar
        info(f"Creating embree instance")
        self.embree = embree.PyEmbreeSolar(
            self.mesh.vertices, self.mesh.faces, self.face_mask
        )
        info(f"Running analysis...")

        # Run raytracing
        if p.sun_analysis:
            if p.sun_approx == SunApprox.quad and sunp.sundome is not None:
                self._sun_quad_raycasting(sunp.sundome, sunp.sunc)
            elif p.sun_approx == SunApprox.group and sunp.sungroups is not None:
                self._sun_group_raycasting(sunp.sungroups, sunp.sunc)
            elif p.sun_approx == SunApprox.none:
                self._sun_raycasting(sunp.sunc)

        if p.sky_analysis:
            self._sky_raycasting()

        # Calculate irradiance base on raytracing results and weather data
        if p.sun_approx == SunApprox.none:
            self.embree.calc_irradiance(sunp.sunc.dni, sunp.sunc.dhi)
        elif p.sun_approx == SunApprox.group:
            indices = sunp.sungroups.sun_indices
            self.embree.calc_irradiance_group(sunp.sunc.dni, sunp.sunc.dhi, indices)
        elif p.sun_approx == SunApprox.quad:
            active_quads = sunp.sundome.active_quads
            indices = [quad.sun_indices for quad in active_quads]
            self.embree.calc_irradiance_group(sunp.sunc.dni, sunp.sunc.dhi, indices)

        # Save results to output collection
        if p.sun_analysis:
            n_suns = sunp.sunc.count
            outc.dni = self.embree.get_results_dni()
            outc.dhi = self.embree.get_results_dhi()
            outc.face_sun_angles = self.embree.get_accumulated_angles() / n_suns
            outc.occlusion = self.embree.get_accumulated_occlusion() / n_suns

        if p.sky_analysis:
            outc.facehit_sky = self.embree.get_face_skyhit_results()

    def _sun_group_raycasting(self, sungroups: SunGroups, sunc: SunCollection):
        sun_vecs = sungroups.list_centers
        info(f"Anlysing {sunc.count} suns in {len(sun_vecs)} sun groups.")
        if self.embree.sun_raytrace_occ8(sun_vecs):
            info(f"Raytracing sun groups completed successfully.")
        else:
            warning(f"Something went wrong in embree solar.")

    def _sun_quad_raycasting(self, sundome: SunDome, sunc: SunCollection):
        sundome.match_suns_and_quads(sunc)
        sun_vecs = sundome.get_active_quad_centers()
        info(f"Anlysing {sunc.count} suns in {len(sun_vecs)} quad groups.")
        if self.embree.sun_raytrace_occ8(sun_vecs):
            info(f"Raytracing sun groups completed successfully.")
        else:
            warning(f"Something went wrong in embree solar.")

    def _sun_raycasting(self, sunc: SunCollection):
        info(f"Running sun raycasting")
        if self.embree.sun_raytrace_occ8(sunc.sun_vecs):
            info(f"Running sun raycasting completed successfully.")
        else:
            warning(f"Something went wrong with sun analysis in embree solar.")

    def _sky_raycasting(self):
        info(f"Running sky raycasting")
        if self.embree.sky_raytrace_occ8():  # Array with nFace elements
            info(f"Running sky raycasting completed succefully.")
        else:
            warning(f"Something went wrong with sky analysis in embree solar.")
