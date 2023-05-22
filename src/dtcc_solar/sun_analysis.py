import numpy as np
import dtcc_solar.raycasting as raycasting
import dtcc_solar.mesh_compute as mc
from dtcc_solar import utils
from dtcc_solar.results import Results
from dtcc_solar.model import Model
from typing import List, Dict
from dtcc_solar.utils import Sun, Res

class SunAnalysis:

    model : Model

    def __init__(self, model):
        self.model = model
        self.flux = 1 #Watts per m2 
        
    def execute_raycasting(self, suns: List[Sun], results:Results):
        n = len(suns)
        print("Iterative analysis started for " + str(n) + " number of iterations")   
        counter = 0

        for sun in suns:
            if sun.over_horizon:
                sun_vec = utils.convert_vec3_to_ndarray(sun.sun_vec)
                sun_vec_rev = utils.reverse_vector(sun_vec)
                
                face_in_sun = raycasting.raytrace_f(self.model, sun_vec_rev)
                face_sun_angles = mc.face_sun_angle(self.model.city_mesh, sun_vec)
                irradianceF = mc.compute_irradiance(face_in_sun, face_sun_angles, self.model.f_count, sun.irradiance_dn)
                 
                results.res_list[sun.index].face_in_sun = face_in_sun
                results.res_list[sun.index].face_sun_angles = face_sun_angles
                results.res_list[sun.index].face_irradiance_dn = irradianceF
                
                counter += 1
                print("Iteration: " + str(counter) + " completed")
