import sys
import pathlib
project_dir = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(project_dir)

import numpy as np
import pandas as pd
import trimesh
import math
import copy

from dtcc_solar import utils

class SkyDome:

    def __init__(self, dome_radius):
        self.meshes_quads = []
        self.meshes_points = []
        self.ray_targets = np.zeros((1000, 3), dtype=np.float64)
        self.ray_areas = np.zeros(1000, dtype=float)
        
        self.joined_mesh_points = 0
        self.dome_mesh = 0
        self.dome_meshes = 0
        self.dome_radius = dome_radius

        self.ray_in_sky = 0
        self.face_in_sky = 0
        self.face_intensity = 0
        self.quad_sun_angle = 0
        self.quad_count = 0
        
        self.create_skydome_mesh()

    def get_quad_count(self):
        return self.quad_count 

    def get_ray_targets(self):
        return self.ray_targets

    def get_ray_areas(self):
        return self.ray_areas
               
    def create_skydome_mesh(self):

        # Assuming the ray origin to be at [0,0,0] so that the sphere of ray targets positions 
        # can be moved easily to each mesh face center for the ray casting.  
              
        n = 10                          
        r = 1                           #Unit radius to be scaled in the intersection calc function 
        r_visual = self.dome_radius                   #Radius for visualisation                      
        top_cap_div = 4                 
        max_azim = 2 * math.pi          
        max_elev = 0.5 * math.pi        
        d_elev = max_elev/n             
        
        elev = max_elev - d_elev
        azim = 0

        #Calculate the area of the initial cap based on the dElev value
        top_cap_area = self.calc_sphere_cap_area(elev, r)
        hemisphere_area = self.calc_hemisphere_area(r)
        target_area = top_cap_area / top_cap_div
        face_area_part = target_area / hemisphere_area

        [self.ray_targets, self.ray_areas, self.meshes_quads, counter] = self.get_top_cap(
                                self.ray_targets,self.ray_areas, max_elev, elev, max_azim, 
                                top_cap_div, r, r_visual, face_area_part, self.meshes_quads)
        
        for i in range(0, n-1):
            #Reducing the elevation for every loop
            next_elev = elev - d_elev
            strip_area = self.calc_sphere_strip_area(elev, next_elev, r)
            n_azim =  int(strip_area / target_area)
            d_azim = max_azim / n_azim
            azim = 0
            face_area_part = (strip_area / n_azim) / hemisphere_area
            mid_pt_elev = (elev + next_elev)/2.0

            for j in range(0, n_azim):
                next_azim = azim + d_azim
                mesh_quad = self.create_dome_mesh_quad(r_visual, azim, next_azim, elev, next_elev)
                mesh_quad.visual.face_colors = [1.0, 0.5, 0, 1.0]
                self.meshes_quads.append(mesh_quad) 

                mid_pt_azim = (azim + next_azim)/2.0
                x = r * math.cos(mid_pt_elev) * math.cos(mid_pt_azim)
                y = r * math.cos(mid_pt_elev) * math.sin(mid_pt_azim)
                z = r * math.sin(mid_pt_elev)
                pt_ray = np.array([x, y, z])

                self.ray_targets[counter, :] = pt_ray
                self.ray_areas[counter] = face_area_part    

                azim += d_azim
                counter += 1
        
            elev = elev - d_elev

        #Remove zeros from the array
        self.ray_targets = self.ray_targets[0:counter, :]
        self.ray_areas = self.ray_areas[0:counter]

        for i in range(0, counter):
            mesh = trimesh.primitives.Sphere(radius = 2.0, center = (r_visual * self.ray_targets[i] ), subdivisions = 1)
            mesh.visual.face_colors = [1.0, 0.5, 0, 1.0]
            self.meshes_points.append(mesh)

        #Join the mesh into one
        self.joined_mesh_points = trimesh.util.concatenate(self.meshes_points)  
        self.dome_mesh = trimesh.util.concatenate(self.meshes_quads) 
        self.quad_count = len(self.meshes_quads)

    def calc_quad_sun_angle(self, sun_vec):
        quad_sun_angle = np.zeros(len(self.ray_targets))
        for i in range(0, len(self.ray_targets)):
            vec_ray = self.ray_targets[i]
            angle = utils.VectorAngle(vec_ray, sun_vec)
            quad_sun_angle[i] = angle

        self.quad_sun_angle = quad_sun_angle


    #Create two triangles that represent the mesh that goes with a particular sky dome ray. 
    def create_dome_mesh_quad(self, ray_length, azim, next_azim, elev, next_elev):
        
        x1 = ray_length * math.cos(elev) * math.cos(azim)
        y1 = ray_length * math.cos(elev) * math.sin(azim)
        z1 = ray_length * math.sin(elev)
        pt1 = np.array([x1,y1,z1])
        
        x2 = ray_length * math.cos(next_elev) * math.cos(azim)
        y2 = ray_length * math.cos(next_elev) * math.sin(azim)
        z2 = ray_length * math.sin(next_elev)
        pt2 = np.array([x2,y2,z2])
        
        x3 = ray_length * math.cos(elev) * math.cos(next_azim)
        y3 = ray_length * math.cos(elev) * math.sin(next_azim)
        z3 = ray_length * math.sin(elev)
        pt3 = np.array([x3,y3,z3])
        
        x4 = ray_length * math.cos(next_elev) * math.cos(next_azim)
        y4 = ray_length * math.cos(next_elev) * math.sin(next_azim)
        z4 = ray_length * math.sin(next_elev)
        pt4 = np.array([x4,y4,z4])

        vs = np.array([pt1, pt2, pt3, pt4])
        fs = np.array([[0,1,2],[1,3,2]])
        mesh = trimesh.Trimesh(vertices = vs, faces = fs)
        
        return mesh

    def create_top_cap_quads(self, ray_length, elev, next_elev, azim, d_azim):

        mid_azim = azim + (d_azim/2)
        next_azim = azim + d_azim

        x1 = ray_length * math.cos(elev) * math.cos(azim)
        y1 = ray_length * math.cos(elev) * math.sin(azim)
        z1 = ray_length * math.sin(elev)
        pt1 = np.array([x1,y1,z1])
        
        x2 = ray_length * math.cos(next_elev) * math.cos(azim)
        y2 = ray_length * math.cos(next_elev) * math.sin(azim)
        z2 = ray_length * math.sin(next_elev)
        pt2 = np.array([x2,y2,z2])
        
        x3 = ray_length * math.cos(next_elev) * math.cos(mid_azim)
        y3 = ray_length * math.cos(next_elev) * math.sin(mid_azim)
        z3 = ray_length * math.sin(next_elev)
        pt3 = np.array([x3,y3,z3])
        
        x4 = ray_length * math.cos(next_elev) * math.cos(next_azim)
        y4 = ray_length * math.cos(next_elev) * math.sin(next_azim)
        z4 = ray_length * math.sin(next_elev)
        pt4 = np.array([x4,y4,z4])

        vs = np.array([pt1, pt2, pt3, pt4])
        fs = np.array([[0,1,2],[2,3,0]])
        
        mesh = trimesh.Trimesh(vertices = vs, faces = fs)

        return mesh

    def get_top_cap(self, ray_target, ray_areas, max_elev, elev, max_azim, n_azim, r, r_visual, face_area_part, meshes_quad):
    
        elev_mid = (elev + max_elev) / 2.0
        d_azim = max_azim / n_azim
        azim_mid = d_azim / 2.0
        counter = 0
        azim = 0

        for i in range(0, n_azim):
            x = r * math.cos(elev_mid) * math.cos(azim_mid)
            y = r * math.cos(elev_mid) * math.sin(azim_mid)
            z = r * math.sin(elev_mid)  
            ray_target[counter,:] = [x ,y ,z]
            ray_areas[counter] = face_area_part

            quad = self.create_top_cap_quads(r_visual, max_elev, elev, azim, d_azim)
            meshes_quad.append(quad)

            azim = azim + d_azim
            azim_mid = azim_mid + d_azim
            counter += 1

        return ray_target, ray_areas, meshes_quad, counter    

    def postprocess_raycasting(self, seg_idxs):
        shaded_portion = np.sum(self.ray_areas[seg_idxs])
        self.ray_in_sky = np.ones(len(self.meshes_quads), dtype=bool) 
        self.face_in_sky = np.ones(2*len(self.meshes_quads), dtype=bool) 
        
        for ray_index in seg_idxs:
            #Set boolean for the ray itself
            self.ray_in_sky[ray_index] = False
            #Set boolean for both quads that represent the ray
            self.face_in_sky[ray_index * 2] = False
            self.face_in_sky[ray_index * 2 + 1] = False

    def move_dome_mesh(self, new_origin):
        new_pos_vec = np.array([0,0,0]) + new_origin
        self.dome_mesh.vertices += new_pos_vec

    def calc_hemisphere_area(self, r):
        area = 2 * math.pi * r * r
        return area

    #Ref: https://www.easycalculation.com/shapes/learn-spherical-cap.php
    def calc_sphere_cap_area(self, elevation, r):
        h = r - r * math.sin(elevation)
        C = 2 * math.sqrt(h * (2 * r - h))
        area = math.pi * (((C * C)/ 4) + h * h) 
        return area

    #Ref: https://www.easycalculation.com/shapes/learn-spherical-cap.php
    def calc_sphere_strip_area(self, elev1, elev2, r):
        h1 = r - r * math.sin(elev1)
        h2 = r - r * math.sin(elev2)
        C1 = 2 * math.sqrt(h1 * (2 * r - h1))
        C2 = 2 * math.sqrt(h2 * (2 * r - h2))
        area1 = math.pi * (((C1 * C1)/ 4) + h1 * h1) 
        area2 = math.pi * (((C2 * C2)/ 4) + h2 * h2)
        return (area2 - area1)













