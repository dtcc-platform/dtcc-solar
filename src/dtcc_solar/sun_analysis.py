import numpy as np
import dtcc_solar.raycasting as raycasting
import dtcc_solar.mesh_compute as mc
from dtcc_solar import utils
from dtcc_solar.results import Results
from dtcc_solar.model import Model
from typing import List, Dict
from dtcc_solar.utils import Sun, Res

class SunAnalysis:

    results : Results
    model : Model

    def __init__(self, model, results):
        self.model = model
        self.results = results 
        self.flux = 1 #Watts per m2
        
    def execute_raycasting_iterative(self, suns: List[Sun], res_list:List[Res]):
        n = len(suns)
        print("Iterative analysis started for " + str(n) + " number of iterations")
        dict_keys = utils.get_dict_keys_from_suns_datetime(suns)
        irradiance_dict = dict.fromkeys(dict_keys)
        face_in_sun_dict = dict.fromkeys(dict_keys)    
        face_sun_angles_dict = dict.fromkeys(dict_keys)    
        counter = 0

        for sun in suns:
            if sun.over_horizon:
                sun_vec = utils.convert_vec3_to_ndarray(sun.sun_vec)
                sun_vec_rev = utils.reverse_vector(sun_vec)
                
                face_in_sun = raycasting.ray_trace_faces(self.model, sun_vec_rev)
                face_sun_angles = mc.sun_face_angle(self.model.city_mesh, sun_vec)
                irradianceF = mc.compute_irradiance(face_in_sun, face_sun_angles, self.model.f_count, sun.irradiance_dn)
                 
                irradiance_dict[sun.datetime_str] = irradianceF
                face_in_sun_dict[sun.datetime_str] = face_in_sun
                face_sun_angles_dict[sun.datetime_str] = face_sun_angles

                res_list[sun.index].face_in_sun = face_in_sun
                res_list[sun.index].face_sun_angle = face_sun_angles
                res_list[sun.index].face_irradiance_dn = irradianceF
                
                counter += 1
                print("Iteration: " + str(counter) + " completed")

        #Register results 
        self.results.set_face_in_sun_dict(face_in_sun_dict)
        self.results.set_face_sun_angles_dict(face_sun_angles_dict)
        self.results.set_sun_irradiance_dict(irradiance_dict)

        self.results.calc_average_results_from_sun_dict(dict_keys)

        return res_list      
    
    def set_city_mesh_out(self):
        self.results.set_city_mesh_out(self.model.city_mesh)