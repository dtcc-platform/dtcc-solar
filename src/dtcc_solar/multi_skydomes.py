import numpy as np
import pandas as pd
import trimesh
import copy

class MultiSkyDomes:

    def __init__(self, skydome):
        self.skydome = 0    
        self.ray_in_sky = 0
        self.face_in_sky = 0        
        self.face_intensity = 0
        self.dome_count = 0
        self.all_dome_meshes = []
        self.nr_quads_per_dome = 0
        self.dome_meshes = 0
        self.skydome = skydome
        self.nr_quads_per_dome = skydome.get_quad_count()
        
    def postprocess_raycasting(self, all_seg_idxs, dome_count, face_mid_pts):
        
        self.dome_count = dome_count
        #Move copy and move the dome to each face mid point that has been identified for evaluation
        for i in range(0, dome_count):
            new_dome_mesh = copy.deepcopy(self.skydome.dome_mesh)
            new_dome_mesh.vertices += face_mid_pts[i]
            self.all_dome_meshes.append(new_dome_mesh)         
        
        self.ray_in_sky = np.ones(dome_count * self.nr_quads_per_dome, dtype=bool) 
        self.face_in_sky = np.ones(dome_count * 2 * self.nr_quads_per_dome, dtype=bool) 
        self.face_intensity = np.zeros(dome_count * 2 * self.nr_quads_per_dome) 
        
        counter = 0
        #Set true and false for rays and faces from ray intersection results
        for dome_index in all_seg_idxs:
            add = dome_index * self.nr_quads_per_dome
            for i in range(0, len(all_seg_idxs[dome_index])):
                shaded_index = all_seg_idxs[dome_index] + add
                self.ray_in_sky[shaded_index] = False    
                self.face_in_sky[shaded_index * 2] = False
                self.face_in_sky[shaded_index * 2 + 1] = False

        counter = 0
        #Add angle from sun for each face at each dome in the mesh.
        for i in range(0, dome_count):
            for j in range(0, self.nr_quads_per_dome):
                if(self.ray_in_sky[counter]):
                    self.face_intensity[counter * 2] = self.skydome.quad_sun_angle[j]
                    self.face_intensity[counter * 2 +1] = self.skydome.quad_sun_angle[j]
                counter += 1            

        # Joining all the dome meshes into one. Global order of the faces 
        # should still match the ray_in_sky and face_in_sky arrays.
        self.dome_meshes = trimesh.util.concatenate(self.all_dome_meshes)     
