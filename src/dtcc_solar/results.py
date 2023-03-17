
import numpy as np
import dtcc_solar.mesh_compute as mc
from dtcc_solar.utils import ColorBy


#This class contains all the results from analysis that will be accessed for visualisation
class Results:

    def __init__(self, city_model):
        
        self.f_count = city_model.f_count
        self.v_count = city_model.v_count

        #Results associated with the city mesh
        self.__city_mesh_out = 0

        #Sun values instant analysis
        self.__face_in_sun = 0                  # Defined as: np.ones(f_count, dtype=bool)
        self.__sun_irradiance = 0               # Defined as: np.zeros(f_count, dtype=float)    
        self.__face_sun_angles = 0              # Defined as: np.zeros(f_count, dtype=float)    

        #Sun values iterative analysis
        self.__face_in_sun_dict = 0             # Defined as: Dict of bools. Key is "Year:month:day h:m:s" i.e "2015:06:30 01:30:00"
        self.__sun_irradiance_dict = 0          # Defined as: Dict of float. Key is "Year:month:day h:m:s" i.e "2015:06:30 01:30:00"
        self.__face_sun_angles_dict = 0         # Defined as: Dict of float. Key is "Year:month:day h:m:s" i.e "2015:06:30 01:30:00"
        
        #Sky values
        self.__face_in_sky = 0                  # Defined as: np.ones(f_count, dtype=bool)
        self.__sky_irradiance = 0               # Defined as: np.zeros(f_count, dtype=float)
        
        #Sky values iterative analysis
        self.__sky_irradiance_dict = 0

        #Results associated with the dome mesh
        self.__dome_mesh_out = 0
        self.__dome_face_in_sky = 0
        self.__dome_sky_irradiance = 0
        
    
    ###################### Getters an setters for city mesh results #######################

    def set_city_mesh_out(self, value):
        self.__city_mesh_out = value 

    def get_city_mesh_out(self):
        return self.__city_mesh_out    

    #######################################################################################

    def set_face_in_sun(self, value):
        self.__face_in_sun = value

    def get_face_in_sun(self):
        return self.__face_in_sun
    
    def set_face_sun_angles(self, value):
        self.__face_sun_angles = value

    def get_face_sun_angles(self):
        return self.__face_sun_angles
    
    def set_sun_irradiance(self, value):
        self.__sun_irradiance = value

    def get_sun_irradiance(self):
        return self.__sun_irradiance    
    
    #######################################################################################                        

    def set_face_sun_angles_dict(self, value):
        self.__face_sun_angles_dict = value

    def get_face_sun_angles_dict(self):
        return self.__face_sun_angles_dict
    
    def set_sun_irradiance_dict(self, value):
        self.__sun_irradiance_dict = value

    def get_sun_irradiance_dict(self):
        return self.__sun_irradiance_dict    
    
    def set_face_in_sun_dict(self, value):
        self.__face_in_sun_dict = value 

    def get_face_in_sun_dict(self):
        return self.__face_in_sun_dict     

    #######################################################################################

    def set_face_in_sky(self, value):
        self.__face_in_sky = value

    def get_face_in_sky(self):
        return self.__face_in_sky
    
    def set_sky_irradiance(self, value):
        self.__sky_irradiance = value    

    def get_sky_irradiance(self):
        return self.__sky_irradiance    

    def set_sky_irradiance_dict(self, value):
        self.__sky_irradiance_dict = value    

    def get_sky_irradiance_dict(self):
        return self.__sky_irradiance_dict    

    def set_color_on_city_mesh(self, face_colors):
        self.__city_mesh_out.visual.face_colors = face_colors

    #######################################################################################

    ###################### Getters an setters for dome mesh results #######################

    def set_dome_mesh_out(self, value):
        self.__dome_mesh_out = value 

    def get_dome_mesh_out(self):
        return self.__dome_mesh_out        

    def set_dome_face_in_sky(self, value):
        self.__dome_face_in_sky = value

    def get_dome_face_in_sky(self):
        return self.__dome_face_in_sky    

    def set_dome_sky_irradiance(self, value):
        self.__dome_sky_irradiance = value

    def get_dome_sky_irradiance(self):
        return self.__dome_sky_irradiance     

    def set_color_on_dome_mesh(self, face_colors):
        self.__dome_mesh_out.visual.face_colors = face_colors

    #######################################################################################

    def color_city_mesh_from_sun(self, color_by):    
        face_colors = []
        #Colors for instant analysis
        if color_by == ColorBy.face_sun_angle:
            mc.calc_face_colors_rayF(self.get_face_sun_angles(), face_colors)
        elif color_by == ColorBy.face_sun_angle_shadows:
            mc.calc_face_with_shadows_colors_rayF(self.get_face_sun_angles(), face_colors, self.get_face_in_sun())
        elif color_by == ColorBy.face_irradiance:
            mc.calc_face_colors_rayF(self.get_sun_irradiance(), face_colors)
        elif color_by == ColorBy.face_shadows:
            mc.calc_face_colors_black_white_rayF(self.get_face_in_sun(), face_colors)
        else:
            print("The selected colorby option is not supported for this type of analysis")    
        
        self.set_color_on_city_mesh(face_colors)

    def color_city_mesh_from_sky(self, color_by):    
        face_colors = []
        #Colors for instant analysis
        if color_by == ColorBy.face_in_sky:
            mc.calc_face_colors_rayF(self.get_face_in_sky, face_colors)
        elif color_by == ColorBy.face_diffusion:
            mc.calc_face_colors_rayF(self.get_sky_irradiance(), face_colors)
        else:
            print("The selected colorby option is not supported for this type of analysis")    
        
        print(len(face_colors))

        self.set_color_on_city_mesh(face_colors)

    def color_dome_mesh(self, color_by):
        face_colors = []
        if color_by == ColorBy.face_in_sky:
            mc.calc_face_colors_dome_face_in_sky(self.get_dome_face_in_sky(), face_colors)
        elif color_by == ColorBy.face_diffusion:
            mc.calc_face_colors_dome_face_intensity(self.get_dome_sky_irradiance(), face_colors)    
        else:
            print("The selected colorby option is not supported for this type of analysis")    
        

        self.set_color_on_dome_mesh(face_colors)

    def color_city_mesh_iterative(self, color_by):

        face_colors = []
        #Colors for iterative analysis    
        if color_by == ColorBy.face_sun_angle:
            mc.calc_face_colors_rayF(self.get_face_sun_angles(), face_colors)
        elif color_by == ColorBy.face_sun_angle_shadows:
            mc.calc_face_with_shadows_colors_rayF(self.get_face_sun_angles(), face_colors, self.get_face_in_sun())
        elif color_by == ColorBy.face_irradiance:
            mc.calc_face_colors_rayF(self.get_sun_irradiance(), face_colors)
        elif color_by == ColorBy.face_shadows:
            mc.calc_face_colors_black_white_rayF(self.get_face_in_sun(), face_colors)
        else:
            print("The selected colorby option is not supported for this type of analysis")    
            
        self.set_color_on_city_mesh(face_colors) 

    def color_city_mesh_com_iterative(self, color_by):

        face_colors = []
        #Colors for iterative analysis    
        if color_by == ColorBy.face_irradiance:
            mc.calc_face_colors_rayF(self.get_sun_irradiance() + self.get_sky_irradiance(), face_colors)
        else:
            print("The selected colorby option is not supported for this type of analysis")    
            
        self.set_color_on_city_mesh(face_colors)     

    def calc_average_results_from_sun_dict(self, dict_keys):
        
        face_count = self.f_count
        avrg_sun_irradiance = np.zeros(face_count)
        avrg_face_in_sun_num = np.zeros(face_count)
        avrg_face_sun_angles = np.zeros(face_count)
        avrg_face_in_sun_bool = np.zeros(face_count, dtype= bool)

        n = len(dict_keys)

        #Calcualte average values for each sun position
        for key in dict_keys:
            for face_index in range(0, face_count):
                irr = self.__sun_irradiance_dict[key][face_index]
                fis = self.__face_in_sun_dict[key][face_index]
                fsa = self.__face_sun_angles_dict[key][face_index]
                avrg_sun_irradiance[face_index] += (irr/n)
                avrg_face_in_sun_num[face_index] += (int(fis)/n)
                avrg_face_sun_angles[face_index] += (fsa/n)    

        avrg_face_in_sun_bool = (avrg_face_in_sun_num > 0.5)

        self.set_face_in_sun(avrg_face_in_sun_bool)
        self.set_sun_irradiance(avrg_sun_irradiance)
        self.set_face_sun_angles(avrg_face_sun_angles)


    def calc_average_results_from_sky_dict(self, dict_keys):
        
        face_count = self.f_count
        avrg_sky_irradiance = np.zeros(face_count)
        
        n = len(dict_keys)

        #Calcualte average values for each sun position
        for key in dict_keys:
            for face_index in range(0, face_count):
                sid = self.__sky_irradiance_dict[key][face_index]
                avrg_sky_irradiance[face_index] += (sid/n)
        
        self.set_sky_irradiance(avrg_sky_irradiance)
        

