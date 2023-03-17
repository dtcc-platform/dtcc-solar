import numpy as np
import trimesh
import copy
import sys
import pathlib
project_dir = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(project_dir)

from dtcc_solar import utils


def compute_irradiance(face_in_sun, face_angles, f_count, flux):
    irradiance = np.zeros(f_count)
    for i in range(0,f_count):
        angle_fraction = face_angles[i] / np.pi         #1 if the angle is pi, which is = 180 degrees. 
        face_in_sun_int = float(face_in_sun[i])
        irradiance[i] = flux * face_in_sun_int * angle_fraction
    
    return irradiance    

def sun_face_angle(mesh, sunVec):
    #Face sun angle calculation
    #vertex_sun_angles = np.zeros(len(mesh.vertices))
    face_sun_angles = np.zeros(len(mesh.faces))
    mesh_faces = list(mesh.faces)
    mesh_face_normals = list(mesh.face_normals)
    for i in range(0, len(mesh_faces)):
        face_normal = mesh_face_normals[i]
        face_sun_angle = 0.0  
        face_sun_angle = utils.VectorAngle(sunVec, face_normal)
        face_sun_angles[i] = face_sun_angle 
        #for vi in mesh_faces[i]:
        #    vertex_sun_angles[vi] += face_sun_angle / 3.0 
    
    return face_sun_angles


def calc_face_with_shadows_colors_rayF(face_sun_angles, face_colors, face_in_sun):    
    min_face_angle = np.min(face_sun_angles[face_in_sun])
    max_face_angle = np.max(face_sun_angles[face_in_sun])
    for i in range(0, len(face_sun_angles)):
        if face_in_sun[i]:
            f_color = utils.GetBlendedColor(min_face_angle, max_face_angle, face_sun_angles[i])
        else:
            f_color = [0.2,0.2,0.2,1] 
        face_colors.append(f_color)

def calc_face_colors_rayF(values, faceColors):    
    min_value = np.min(values)
    max_value = np.max(values)
    for i in range(0, len(values)):
        fColor = utils.GetBlendedColor(min_value, max_value, values[i]) 
        faceColors.append(fColor)

def calc_face_colors_black_white_rayF(values, faceColors):    
    max_value = np.max(values)
    for i in range(0, len(values)):
        fColor = utils.GetBlendedColorBlackAndWhite(max_value, values[i]) 
        faceColors.append(fColor)

def calc_face_colors_dome_face_in_sky(values, faceColors):    
    max_value = np.max(values)
    for i in range(0, len(values)):
        fColor = utils.GetBlendedColorRedAndBlue(max_value, values[i]) 
        faceColors.append(fColor)        

def calc_face_colors_dome_face_intensity(values, faceColors):    
    max_value = np.max(values)
    min_value = np.min(values)
    for i in range(0, len(values)):
        fColor = utils.GetBlendedColor(min_value, max_value, values[i])
        faceColors.append(fColor)

def find_shadow_border_faces_rayV(mesh, faceShading):
    borderFaceMask = np.ones(len(mesh.faces), dtype = bool)
    faces = list(mesh.faces)
    for i in range(len(faces)):
        if faceShading[i] < 3 and faceShading[i] > 0 :
            borderFaceMask[i] = False
    
    return borderFaceMask      


def find_shadow_borded_faces_rayF(mesh, faceShading):
    borderFaceMask = np.ones(len(mesh.faces), dtype = bool)
    faces = list(mesh.faces)
    for i in range(len(faces)):
        if faceShading[i] < 3 and faceShading[i] > 0 :
            borderFaceMask[i] = False
    
    return borderFaceMask


def split_mesh(mesh, borderFaceMask, faceShading, face_in_sun):
    #Reversed face mask booleans 
    borderFaceMask_not = [not elem for elem in borderFaceMask]
    
    meshNormal = copy.deepcopy(mesh)
    meshNormal.update_faces(borderFaceMask)
    meshNormal.remove_unreferenced_vertices()

    face_shading_normal = faceShading[borderFaceMask]
    face_in_sun_normal = face_in_sun[borderFaceMask]

    meshborder = copy.deepcopy(mesh)
    meshborder.update_faces(borderFaceMask_not)
    meshborder.remove_unreferenced_vertices()
    return [meshNormal, meshborder, face_shading_normal, face_in_sun_normal]
    
def subdivide_border(meshBorder, maxEdgeLength, maxIter):
    [vs, fs] = trimesh.remesh.subdivide_to_size(meshBorder.vertices, meshBorder.faces, max_edge = maxEdgeLength, max_iter = maxIter, return_index = False)
    meshBorderSD = trimesh.Trimesh(vs, fs)
    return meshBorderSD


def calculate_average_edge_length(mesh):
    edges = mesh.edges_unique
    eCount = len(edges)
    vertices = list(mesh.vertices)
    edgeL = 0

    for edge in edges:
        vIndex1 = edge[0]
        vIndex2 = edge[1]
        d = utils.Distance(vertices[vIndex1], vertices[vIndex2])
        edgeL += d

    edgeL = edgeL / eCount

    return edgeL
