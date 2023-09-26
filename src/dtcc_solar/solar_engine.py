import math
import numpy as np
import dtcc_solar.raycasting as raycasting
import trimesh
import copy
import dtcc_solar.mesh_compute as mc
import dtcc_solar.utils as utils
from ncollpyde import Volume
from dtcc_solar.results import Results
from dtcc_solar.skydome import SkyDome
from typing import List, Any
from dtcc_solar.utils import Sun


class SolarEngine:
    mesh: trimesh
    origin: np.ndarray
    f_count: int
    v_count: int
    horizon_z: float
    sunpath_radius: float
    sun_size: float
    bb: np.ndarray
    bbx: np.ndarray
    bby: np.ndarray
    bbz: np.ndarray
    volume: Volume
    city_mesh_faces: np.ndarray
    city_mesh_points: np.ndarray
    city_mesh_face_mid_points: np.ndarray
    skydome: SkyDome

    def __init__(self, mesh: trimesh) -> None:
        self.mesh = mesh
        self.origin = np.array([0, 0, 0])
        self.f_count = len(self.mesh.faces)
        self.v_count = len(self.mesh.vertices)

        self.horizon_z = 0
        self.sunpath_radius = 0
        self.sun_size = 0
        self.dome_radius = 0
        self.bb = 0
        self.bbx = 0  # (min_x, max_x)
        self.bby = 0  # (min_y, max_y)
        self.bbz = 0  # (min_z, max_z)
        self.preprocess_mesh(True)

        # Create volume object for ray caster with NcollPyDe
        self.volume = Volume(self.mesh.vertices, self.mesh.faces)
        self.city_mesh_faces = np.array(self.volume.faces)
        self.city_mesh_points = np.array(self.volume.points)
        self.city_mesh_face_mid_points = 0

        self.skydome = SkyDome(self.dome_radius)

    def preprocess_mesh(self, center_mesh):
        bb = trimesh.bounds.corners(self.mesh.bounding_box.bounds)
        bbx = np.array([np.min(bb[:, 0]), np.max(bb[:, 0])])
        bby = np.array([np.min(bb[:, 1]), np.max(bb[:, 1])])
        bbz = np.array([np.min(bb[:, 2]), np.max(bb[:, 2])])
        # Center mesh based on x and y coordinates only
        centerVec = self.origin - np.array([np.average([bbx]), np.average([bby]), 0])

        # Move the mesh to the centre of the model
        if center_mesh:
            self.mesh.vertices += centerVec

        self.horizon_z = np.average(self.mesh.vertices[:, 2])

        # Update bounding box after the mesh has been moved
        self.bb = trimesh.bounds.corners(self.mesh.bounding_box.bounds)
        self.bbx = np.array([np.min(self.bb[:, 0]), np.max(self.bb[:, 0])])
        self.bby = np.array([np.min(self.bb[:, 1]), np.max(self.bb[:, 1])])
        self.bbz = np.array([np.min(self.bb[:, 2]), np.max(self.bb[:, 2])])

        # Calculating sunpath radius
        xRange = np.max(self.bb[:, 0]) - np.min(self.bb[:, 0])
        yRange = np.max(self.bb[:, 1]) - np.min(self.bb[:, 1])
        zRange = np.max(self.bb[:, 2]) - np.min(self.bb[:, 2])
        self.sunpath_radius = math.sqrt(
            math.pow(xRange / 2, 2) + math.pow(yRange / 2, 2) + math.pow(zRange / 2, 2)
        )
        self.sun_size = self.sunpath_radius / 90.0
        self.dome_radius = self.sunpath_radius / 40

    def sun_raycasting(self, suns: List[Sun], results: Results):
        n = len(suns)
        print("Iterative analysis started for " + str(n) + " number of iterations")
        counter = 0

        print(results)

        for sun in suns:
            if sun.over_horizon:
                sun_vec = utils.convert_vec3_to_ndarray(sun.sun_vec)
                sun_vec_rev = utils.reverse_vector(sun_vec)

                face_in_sun = raycasting.raytrace_f(self.volume, sun_vec_rev)
                face_sun_angles = mc.face_sun_angle(self.mesh, sun_vec)
                irradianceF = mc.compute_irradiance(
                    face_in_sun, face_sun_angles, self.f_count, sun.irradiance_dn
                )

                results.res_list[sun.index].face_in_sun = face_in_sun
                results.res_list[sun.index].face_sun_angles = face_sun_angles
                results.res_list[sun.index].face_irradiance_dn = irradianceF

                counter += 1
                print("Iteration: " + str(counter) + " completed")

    def sky_raycasting(self, suns: List[Sun], results: Results):
        sky_portion = raycasting.raytrace_skydome(
            self.volume, self.skydome.get_ray_targets(), self.skydome.get_ray_areas()
        )

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
