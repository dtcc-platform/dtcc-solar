import numpy as np
import trimesh
import dtcc_solar.mesh_compute as mc
from dtcc_solar import utils

class Viewer:

    def __init__(self):
        self.scene = trimesh.Scene()
        self.scene.camera._fov = [35,35]     #Field of view [x, y]
        self.scene.camera.z_far = 10000      #Distance to the far clipping plane        

    def add_meshes(self, meshes):
        self.scene.add_geometry(meshes)

    def add_dome_mesh(self, dome_mesh):
        self.scene.add_geometry(dome_mesh)            

    def show(self):
        self.scene.show()               



class Colors:

    def __init__(self):
        pass

    def calc_colors(values):
        colors = []    
        min_value = np.min(values)
        max_value = np.max(values)
        for i in range(0, len(values)):
            c = utils.get_blended_color(min_value, max_value, values[i]) 
            colors.append(c)
        return colors    

    # Calculate color blend for a range of values where some are excluded using a True-False mask
    def calc_colors_with_mask(values, mask):    
        colors = []
        min = np.min(values[mask])
        max = np.max(values[mask])
        for i in range(0, len(values)):
            if mask[i]:
                c = utils.get_blended_color(min, max, values[i])
            else:
                c = [0.2,0.2,0.2,1] 
            colors.append(c)
        return colors    
    
    # Calculate bleded color in a monochrome scale
    def calc_colors_mono(values):
        colors = []    
        max_value = np.max(values)
        for i in range(0, len(values)):
            fColor = utils.get_blended_color_mono(max_value, values[i]) 
            colors.append(fColor)
