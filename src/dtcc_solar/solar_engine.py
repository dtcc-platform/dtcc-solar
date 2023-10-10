import math
import numpy as np
import dtcc_solar.raycasting as ray
import trimesh
import copy
import dtcc_solar.mesh_compute as mc
import dtcc_solar.utils as utils
from ncollpyde import Volume
from dtcc_solar.results import Results
from dtcc_solar.skydome import SkyDome
from typing import List, Any
from dtcc_solar.utils import Sun
from trimesh import Trimesh

from dtcc_model import Mesh
from dtcc_model.geometry import Bounds


class SolarEngine:
    dmesh: Mesh  # Dtcc mesh
    tmesh: Trimesh  # Trimesh
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
    volume: Volume
    city_mesh_faces: np.ndarray
    city_mesh_points: np.ndarray
    city_mesh_face_mid_points: np.ndarray
    skydome: SkyDome

    def __init__(self, dmesh: Mesh) -> None:
        self.dmesh = dmesh
        self.tmesh = trimesh.Trimesh(vertices=dmesh.vertices, faces=dmesh.faces)
        self.origin = np.array([0, 0, 0])
        self.f_count = len(self.dmesh.faces)
        self.v_count = len(self.dmesh.vertices)

        self.horizon_z = 0
        self.sunpath_radius = 0
        self.sun_size = 0
        self.dome_radius = 0
        self.path_width = 0
        self.preprocess_mesh(True)

        # Create volume object for ray caster with NcollPyDe
        self.volume = Volume(self.dmesh.vertices, self.dmesh.faces)
        self.city_mesh_faces = np.array(self.volume.faces)
        self.city_mesh_points = np.array(self.volume.points)
        self.city_mesh_face_mid_points = 0

        self.skydome = SkyDome(self.dome_radius)

    def preprocess_mesh(self, move_to_center: bool):
        self._calc_bounds()
        # Center mesh based on x and y coordinates only
        center_bb = np.array([np.average(self.bbx), np.average(self.bby), 0])
        centerVec = self.origin - center_bb

        # Move the mesh to the centre of the model
        if move_to_center:
            self.dmesh.vertices += centerVec
            self.tmesh.vertices += centerVec

        # Update bounding box after the mesh has been moved
        self._calc_bounds()

        # Assumption: The horizon is the avrage z height in the mesh
        self.horizon_z = np.average(self.dmesh.vertices[:, 2])

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
        self.path_width = self.sunpath_radius / 300

    def _calc_bounds(self):
        self.xmin = self.dmesh.vertices[:, 0].min()
        self.xmax = self.dmesh.vertices[:, 0].max()
        self.ymin = self.dmesh.vertices[:, 1].min()
        self.ymax = self.dmesh.vertices[:, 1].max()
        self.zmin = self.dmesh.vertices[:, 2].min()
        self.zmax = self.dmesh.vertices[:, 2].max()

        self.bbx = np.array([self.xmin, self.xmax])
        self.bby = np.array([self.ymin, self.ymax])
        self.bbz = np.array([self.zmin, self.zmax])

        self.bb = Bounds(xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax)

    def sun_precasting(self, suns: List[Sun], results: Results):
        pass

    def sun_raycasting(self, suns: List[Sun], results: Results):
        n = len(suns)
        print("Iterative analysis started for " + str(n) + " number of iterations")
        counter = 0

        for sun in suns:
            if sun.over_horizon:
                sun_vec = utils.vec_2_ndarray(sun.sun_vec)
                sun_vec_rev = utils.reverse_vector(sun_vec)

                face_in_sun = ray.raytrace(self.volume, sun_vec_rev)
                face_sun_angles = mc.face_sun_angle(self.tmesh, sun_vec)
                irradianceF = mc.compute_irradiance(
                    face_in_sun, face_sun_angles, self.f_count, sun.irradiance_dn
                )

                results.res_list[sun.index].face_in_sun = face_in_sun
                results.res_list[sun.index].face_sun_angles = face_sun_angles
                results.res_list[sun.index].face_irradiance_dn = irradianceF

                counter += 1
                print("Iteration: " + str(counter) + " completed")

    def sky_raycasting(self, suns: List[Sun], results: Results):
        ray_targets = self.skydome.get_ray_targets()
        ray_areas = self.skydome.get_ray_areas()
        sky_portion = ray.raytrace_skydome(self.volume, ray_targets, ray_areas)

        # Results independent of weather data
        face_in_sky = np.ones(len(sky_portion), dtype=bool)
        for i in range(0, len(sky_portion)):
            if sky_portion[i] > 0.5:
                face_in_sky[i] = False

        results.res_acum.face_in_sky = face_in_sky

        # Results which depends on wheater data
        for sun in suns:
            irradiance_diffuse = sun.irradiance_di
            sky_portion_copy = copy.deepcopy(sky_portion)
            diffuse_irradiance = irradiance_diffuse * sky_portion_copy
            results.res_list[sun.index].face_irradiance_di = diffuse_irradiance
