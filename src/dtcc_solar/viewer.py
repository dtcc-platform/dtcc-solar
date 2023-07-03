import numpy as np
import trimesh
import dtcc_solar.mesh_compute as mc
from dtcc_solar import utils
from dtcc_solar.utils import ColorBy, AnalysisType, Output
from typing import List, Dict

class Viewer:

    def __init__(self):
        
        self.scene = trimesh.Scene()
        self.scene.camera._fov = [35,35]     #Field of view [x, y]
        self.scene.camera.z_far = 10000      #Distance to the far clipping plane        

    def add_meshes(self, meshes):
        self.scene.add_geometry(meshes)       

    def show(self):
        self.scene.show()               


class Colors:

    def __init__(self):
        pass

    def color_city_mesh(self, mesh, res:Output, color_by:ColorBy):
        if  (color_by == ColorBy.face_sun_angle):
            colors = self.calc_colors(res.face_sun_angles)
        elif(color_by == ColorBy.face_sun_angle_shadows):
            colors = self.calc_colors_with_mask(res.face_sun_angles, res.face_in_sun)
        elif(color_by == ColorBy.face_irradiance_dn):
            colors = self.calc_colors(res.face_irradiance_dn)
        elif(color_by == ColorBy.face_irradiance_di):
            colors = self.calc_colors(res.face_irradiance_di)
        elif(color_by == ColorBy.face_irradiance_tot):
            colors = self.calc_colors(res.face_irradiance_tot)    
        else:
            print("Color calculation for city mesh failed!")    

        mesh.visual.face_colors = colors

    def color_analemmas():
        pass


    def calc_colors(self, values):
        colors = []    
        min_value = np.min(values)
        max_value = np.max(values)
        for i in range(0, len(values)):
            c = self.get_blended_color(min_value, max_value, values[i]) 
            colors.append(c)
        return colors    

    # Calculate color blend for a range of values where some are excluded using a True-False mask
    def calc_colors_with_mask(self, values, mask):    
        colors = []
        min = np.min(values[mask])
        max = np.max(values[mask])
        for i in range(0, len(values)):
            if mask[i]:
                c = self.get_blended_color(min, max, values[i])
            else:
                c = [0.2,0.2,0.2,1] 
            colors.append(c)
        return colors    

    # Calculate bleded color in a monochrome scale
    def calc_colors_mono(self, values):
        colors = []    
        max_value = np.max(values)
        for i in range(0, len(values)):
            fColor = self.get_blended_color_mono(max_value, values[i]) 
            colors.append(fColor)

    # Color mesh for domes that can be used for debugging
    def color_dome(self, values, faceColors):    
        max_value = np.max(values)
        min_value = np.min(values)
        for i in range(0, len(values)):
            fColor = self.get_blended_color(min_value, max_value, values[i])
            faceColors.append(fColor)

    def get_blended_color(self, min, max, value):
        diff = max - min
        newMax = diff
        newValue = value - min
        percentage = 100.0 * (newValue / newMax)

        if (percentage >= 0.0 and percentage <= 25.0):
            #Blue fading to Cyan [0,x,255], where x is increasing from 0 to 255
            frac = percentage / 25.0
            return [0.0, (frac * 1.0), 1.0 , 1.0]

        elif (percentage > 25.0 and percentage <= 50.0):
            #Cyan fading to Green [0,255,x], where x is decreasing from 255 to 0
            frac = 1.0 - abs(percentage - 25.0) / 25.0
            return [0.0, 1.0, (frac * 1.0), 1.0]

        elif (percentage > 50.0 and percentage <= 75.0):
            #Green fading to Yellow [x,255,0], where x is increasing from 0 to 255
            frac = abs(percentage - 50.0) / 25.0
            return [(frac * 1.0), 1.0, 0.0, 1.0 ]

        elif (percentage > 75.0 and percentage <= 100.0):
            #Yellow fading to red [255,x,0], where x is decreasing from 255 to 0
            frac = 1.0 - abs(percentage - 75.0) / 25.0
            return [1.0, (frac * 1.0), 0.0, 1.0]

        elif (percentage > 100.0):
            #Returning red if the value overshoot the limit.
            return [1.0, 0.0, 0.0, 1.0 ]

        return [0.5, 0.5, 0.5, 1.0 ]


    def get_blended_color_mono(self, max, value):
        frac = 0
        if(max > 0):
            frac = value / max
        return [frac, frac, frac, 1.0]

    def get_blended_color_red_blue(self, max, value):
        frac = 0
        if(max > 0):
            frac = value / max
        return [frac, 0.0, 1 - frac, 1.0]

    def get_blended_color_yellow_red(self, max, value):
        percentage = 100.0 * (value / max)
        if (value < 0):
            return [1.0, 1.0, 1.0, 1.0]
        else:
            # Yellow [255, 255, 0] fading to red [255, 0, 0] 
            frac = 1 - percentage/100
            return [1.0, (frac * 1.0), 0.0, 1.0]
