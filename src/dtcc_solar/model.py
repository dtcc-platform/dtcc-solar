import numpy as np
import math
import trimesh
from ncollpyde import Volume

class Model:

    def __init__(self, mesh) -> None:
        self.city_mesh = mesh
        self.origin = np.array([0, 0, 0])
        self.f_count = len(self.city_mesh.faces)
        self.v_count = len(self.city_mesh.vertices) 

        self.horizon_z = 0
        self.sunpath_radius = 0
        self.sun_size = 0
        self.dome_radius = 0
        self.bb = 0                 
        self.bbx = 0                #(min_x, max_x)
        self.bby = 0                #(min_y, max_y)
        self.bbz = 0                #(min_z, max_z)
        self.preprocess_mesh(True)

        #Create volume object for ray caster with NcollPyDe
        self.volume = Volume(self.city_mesh.vertices, self.city_mesh.faces)
        self.city_mesh_faces = np.array(self.volume.faces)
        self.city_mesh_points = np.array(self.volume.points)
        
        #The output mesh might have changed with subdee etc
        self.city_mesh_out = 0                                  
        self.dome_mesh_out = 0

        self.city_mesh_face_mid_points = 0

    def set_dome_mesh_out(self, dome_mesh):
        self.dome_mesh_out = dome_mesh    

    def set_city_mesh_out(self, city_mesh):
        self.city_mesh_out = city_mesh 

    def set_city_mesh_face_mid_points(self, pts):
        self.city_mesh_face_mid_points = pts    

    def preprocess_mesh(self, center_mesh):
        bb = trimesh.bounds.corners(self.city_mesh.bounding_box.bounds)
        bbx = np.array([np.min(bb[:,0]), np.max(bb[:,0])])
        bby = np.array([np.min(bb[:,1]), np.max(bb[:,1])])
        bbz = np.array([np.min(bb[:,2]), np.max(bb[:,2])])
        #Center mesh based on x and y coordinates only
        centerVec = self.origin - np.array([np.average([bbx]), np.average([bby]), 0])

        #Move the mesh to the centre of the model
        if center_mesh:
            self.city_mesh.vertices += centerVec

        self.horizon_z = np.average(self.city_mesh.vertices[:,2])    
        
        #Update bounding box after the mesh has been moved
        self.bb = trimesh.bounds.corners(self.city_mesh.bounding_box.bounds)
        self.bbx = np.array([np.min(self.bb[:,0]), np.max(self.bb[:,0])])
        self.bby = np.array([np.min(self.bb[:,1]), np.max(self.bb[:,1])])
        self.bbz = np.array([np.min(self.bb[:,2]), np.max(self.bb[:,2])])

        #Calculating sunpath radius
        xRange = np.max(self.bb[:,0]) - np.min(self.bb[:,0])
        yRange = np.max(self.bb[:,1]) - np.min(self.bb[:,1])
        zRange = np.max(self.bb[:,2]) - np.min(self.bb[:,2])
        self.sunpath_radius = math.sqrt( math.pow(xRange/2, 2) + math.pow(yRange/2, 2) + math.pow(zRange/2, 2) )
        self.sun_size = self.sunpath_radius / 90.0
        self.dome_radius = self.sunpath_radius / 40

    def calc_city_mesh_face_mid_points(self):

        faceVertexIndex1 = self.city_mesh_faces[:,0]
        faceVertexIndex2 = self.city_mesh_faces[:,1]
        faceVertexIndex3 = self.city_mesh_faces[:,2] 
    
        vertex1 = self.city_mesh_points[faceVertexIndex1]
        vertex2 = self.city_mesh_points[faceVertexIndex2]
        vertex3 = self.city_mesh_points[faceVertexIndex3]

        self.city_mesh_face_mid_points = (vertex1 + vertex2 + vertex3)/3.0
        
    def refine_city_mesh(self, sun_vec):
        pass

