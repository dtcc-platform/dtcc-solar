import numpy as np
import dtcc_solar.raycasting as raycasting
import trimesh
import copy
import dtcc_solar.mesh_compute as mc
import dtcc_solar.utils as utils

from dtcc_solar.results import Results
from dtcc_solar.skydome import SkyDome
from dtcc_solar.model import Model
from dtcc_solar.multi_skydomes import MultiSkyDomes

from typing import List, Dict
from dtcc_solar.utils import Sun, Res


class SkyAnalysis: 

    skydome : SkyDome
    model : Model
    multi_skydomes : MultiSkyDomes 

    def __init__(self, model):
        self.model = model
        self.skydome = SkyDome(self.model.dome_radius)
        self.multi_skydomes = MultiSkyDomes(self.skydome)

    def execute_raycasting_domes(self, suns:List[Sun]):
        face_indices_grid = self.find_distributed_face_mid_points(10,12) #generate a grid of points for dome evaluation
        [all_seg_idxs, face_mid_pts] = raycasting.ray_trace_sky_some(self.model, self.skydome.get_ray_targets(), face_indices_grid)
        self.skydome.calc_quad_sun_angle(utils.convert_vec3_to_ndarray(suns[0].sun_vec))
        self.multi_skydomes.postprocess_raycasting(all_seg_idxs, len(face_indices_grid), face_mid_pts)

    def execute_raycasting_iterative(self, suns:List[Sun], results:Results):
        sky_portion = raycasting.ray_trace_sky(self.model, self.skydome.get_ray_targets(), 
                                               self.skydome.get_ray_areas())
        
        # Results independent of weather data
        face_in_sky = np.ones(len(sky_portion), dtype=bool) 
        for i in range(0, len(sky_portion)):
            if(sky_portion[i] > 0.5):
                face_in_sky[i] = False

        results.res_acum.face_in_sky = face_in_sky        
        
        # Results which depends on wheater data
        for sun in suns:
            irradiance_diffuse =  sun.irradiance_di
            sky_portion_copy = copy.deepcopy(sky_portion)
            diffuse_irradiance = irradiance_diffuse * sky_portion_copy
            results.res_list[sun.index].face_irradiance_di = diffuse_irradiance
        
    def find_distributed_face_mid_points(self, nx, ny):

        dx = (self.model.bbx[1] - self.model.bbx[0])/(nx-1)
        dy = (self.model.bby[1] - self.model.bby[0])/(ny-1)

        x = self.model.bbx[0]
        y = self.model.bby[0]
        z = np.average([self.model.bbz])

        self.model.calc_city_mesh_face_mid_points()
        face_mid_points = self.model.city_mesh_face_mid_points
        face_mid_points[:,2] = z 

        face_indices = []

        for i in range(0,nx):
            y = self.model.bby[0]
            for j in range(0,ny):
                point = np.array([x,y,z])
                index = utils.get_index_of_closest_point(point, face_mid_points)
                face_indices.append(index)
                y += dy
            x += dx

        return face_indices        
























