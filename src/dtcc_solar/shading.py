import copy
import trimesh
import numpy as np

import dtcc_solar.raycasting as raycasting
import dtcc_solar.mesh_compute as mc
from dtcc_solar import utils


class Shading:

    def __init__(self, meshes):
        self.meshes = meshes
        self.mesh_collection = []
        self.faceShadingDict = {}
        self.faceAnglesDict = {}
        self.vertex_angles_dict = {}
        self.face_colors_dict = {}
        self.flux = 1000 #Watts per m2

        self.face_in_sun = 0
        self.face_sun_angles = 0
        self.face_irradiance = 0
        self.face_in_sun_sum = 0

    def ExecuteShadingV(self, sun_vec):    
        iterCount = 10
        maxIter = 5
        edgeLengthAverage = mc.calculate_average_edge_length(self.meshes.city_mesh)
        targetLength = 0.05 * edgeLengthAverage
        evalMesh = copy.deepcopy(self.meshes.city_mesh)
        sun_vec_rev = utils.reverse_vector(sun_vec)
        all_face_in_sun = np.array([], dtype=bool)
        all_face_shading = np.array([])

        for i in range(0,iterCount):
            maxEdgeLength = (1 - i/iterCount) * (edgeLengthAverage - targetLength) + targetLength       
            [faceShading, vertexInSun, face_in_sun]  = raycasting.ray_trace_V(evalMesh, self.meshes.volume, sun_vec_rev)
            borderFaceMask = mc.find_shadow_border_faces_rayV(evalMesh, faceShading)
            [meshNormal, meshBorder, face_shading_normal, face_in_sun_normal] = mc.split_mesh(evalMesh, borderFaceMask, faceShading, face_in_sun)   
            all_face_in_sun  = np.append(all_face_in_sun, face_in_sun_normal)
            mesh_border_SD = mc.subdivide_border(meshBorder, maxEdgeLength, maxIter)
            evalMesh = mesh_border_SD     #Set the subDee mesh to be evaluated next iteration!
            self.mesh_collection.append(meshNormal)
            all_face_shading = np.append(all_face_shading, face_shading_normal.tolist())
            
        #Calcuate the face shading for the finest Subdee level of the triangles 
        [faceShadingSD, vertexInSun, face_in_sun]  = raycasting.ray_trace_V(mesh_border_SD, self.meshes.volume, sun_vec_rev)
        all_face_in_sun  = np.append(all_face_in_sun, face_in_sun)
        self.mesh_collection.append(mesh_border_SD)     #Add the subDee mesh as the last step!    
        all_face_shading = np.append(all_face_shading, faceShadingSD.tolist())
        
        joined_mesh = trimesh.util.concatenate(self.mesh_collection)    

        [all_face_sun_angles, vertex_angles] = mc.face_sun_angle(joined_mesh, sun_vec)
        all_face_irradiance = mc.compute_irradiance(all_face_in_sun, all_face_sun_angles, self.flux, len(joined_mesh.faces))

        self.face_in_sun = all_face_in_sun
        self.face_sun_angles = all_face_sun_angles
        self.face_irradiance = all_face_irradiance

        self.meshes.set_city_mesh_out(joined_mesh)





