import numpy as np
import dtcc_solar.raycasting as raycasting
import dtcc_solar.mesh_compute as mc
from dtcc_solar import utils
from dtcc_solar.results import Results
from dtcc_solar.model import Model

class SunAnalysis:

    results : Results
    model : Model

    def __init__(self, model, results):
        self.model = model
        self.results = results 
        self.flux = 1 #Watts per m2
        
    def execute_raycasting(self, sun_vec):    
        sun_vec_rev = utils.reverse_vector(sun_vec)
        face_in_sun = raycasting.ray_trace_faces(self.model, sun_vec_rev)
        face_sun_angles = mc.sun_face_angle(self.model.city_mesh, sun_vec)
        sun_irradiance = mc.compute_irradiance(face_in_sun, face_sun_angles, self.model.f_count, self.flux)
        
        #Register results 
        self.results.set_face_in_sun(face_in_sun)
        self.results.set_face_sun_angles(face_sun_angles)
        self.results.set_sun_irradiance(sun_irradiance)      
    
    def execute_raycasting_iterative(self, sun_vecs_dict, dict_keys, w_data):
        n = len(sun_vecs_dict)
        print("Iterative analysis started for " + str(n) + " number of iterations")
        all_irradiance = dict.fromkeys(dict_keys)
        all_face_in_sun = dict.fromkeys(dict_keys)    
        all_face_sun_angles = dict.fromkeys(dict_keys)    
        counter = 0

        for key in dict_keys:
            sun_vec = sun_vecs_dict[key]
            sun_vec_rev = utils.reverse_vector(sun_vec)
            face_in_sun = raycasting.ray_trace_faces(self.model, sun_vec_rev)
            face_sun_angles = mc.sun_face_angle(self.model.city_mesh, sun_vec)
            flux = w_data[key]['normal_irradiance']
            irradianceF = mc.compute_irradiance(face_in_sun, face_sun_angles, self.model.f_count, flux)
            
            all_irradiance[key] = irradianceF
            all_face_in_sun[key] = face_in_sun
            all_face_sun_angles[key] = face_sun_angles
            
            counter += 1
            print("Iteration: " + str(counter) + " completed")

        #Register results 
        self.results.set_face_in_sun_dict(all_face_in_sun)
        self.results.set_face_sun_angles_dict(all_face_sun_angles)
        self.results.set_sun_irradiance_dict(all_irradiance)

        self.results.calc_average_results_from_sun_dict(dict_keys)      
    
    def set_city_mesh_out(self):
        self.results.set_city_mesh_out(self.model.city_mesh)