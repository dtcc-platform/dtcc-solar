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


class SkyAnalysis: 

    results : Results
    skydome : SkyDome
    model : Model
    multi_skydomes : MultiSkyDomes 

    def __init__(self, model, results, skydome, multi_skydomes):
        self.model = model
        self.results = results
        self.skydome = skydome
        self.multi_skydomes = multi_skydomes
    
    def execute_raycasting(self, sun_vec, dict_keys, w_data):
        sky_portion = raycasting.ray_trace_sky( self.model, 
                                                    self.skydome.get_ray_targets(), 
                                                    self.skydome.get_ray_areas(),
                                                    w_data)


        [sky_irradiance_dict, face_in_sky] = self.postprocess_sky(sky_portion, dict_keys, w_data)
    
        #Register results
        self.results.set_face_in_sky(face_in_sky)
        self.results.set_sky_irradiance_dict(sky_irradiance_dict)
        self.results.calc_average_results_from_sky_dict(dict_keys)

    def execute_raycasting_some(self, sun_vec):
        face_indices_grid = self.find_distributed_face_mid_points(10,12) #generate a grid of points for dome evaluation
        [all_seg_idxs, face_mid_pts] = raycasting.ray_trace_sky_some(self.model, 
                                                                     self.skydome.get_ray_targets(), 
                                                                     face_indices_grid,)
        self.skydome.calc_quad_sun_angle(sun_vec)
        self.multi_skydomes.postprocess_raycasting(all_seg_idxs, len(face_indices_grid), face_mid_pts)
        
        #Register results
        self.results.set_dome_face_in_sky(self.multi_skydomes.face_in_sky) 
        self.results.set_dome_sky_irradiance(self.multi_skydomes.face_intensity) 

    def set_city_mesh_out(self):
        self.results.set_city_mesh_out(self.model.city_mesh)
        
    def set_dome_mesh_out(self):
        self.results.set_dome_mesh_out(self.multi_skydomes.dome_meshes)
        
    def postprocess_sky(self, sky_portion, dict_keys, w_data):
        face_in_sky = np.ones(len(sky_portion), dtype=bool) 
        for i in range(0, len(sky_portion)):
            if(sky_portion[i] > 0.5):
                face_in_sky[i] = False
        
        sky_irradiance_dict = dict.fromkeys(dict_keys)
        for key in dict_keys:
            flux = w_data[key]['diffuse_radiation']
            sky_portion_copy = copy.deepcopy(sky_portion)
            sky_irradiance_dict[key] = flux * sky_portion_copy
        
        return sky_irradiance_dict, face_in_sky

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
























