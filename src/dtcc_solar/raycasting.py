#Raycasting function using ncollpyde 
import numpy as np
import time
from dtcc_solar import utils
from ncollpyde import Volume

def raytrace_f(volume:Volume, sunVecRev):
    
    mesh_faces = volume.faces
    mesh_points = volume.points
    fCount = len(mesh_faces)
        
    [ptRayOrigin, ptRayTarget] = pre_process_f(mesh_faces, mesh_points, sunVecRev)
    [seg_idxs, intersections, is_backface] = volume.intersections(ptRayOrigin, ptRayTarget,)
    face_in_sun = post_process_f(seg_idxs, fCount)
    
    #print("---- Face midpoint intersection results ----")
    print("Found nr of intersections: " + str(len(seg_idxs)))

    return face_in_sun
    
def pre_process_f(mesh_faces, mesh_points, sun_vec_rev):
    
    ptRayOrigin = np.zeros([len(mesh_faces), 3])
    ptRayTarget = np.zeros([len(mesh_faces), 3])
    tol = 0.01
    rayLength = 1000.0

    sunVecRevNp = np.array(sun_vec_rev)
    faceVertexIndex1 = mesh_faces[:,0]
    faceVertexIndex2 = mesh_faces[:,1]
    faceVertexIndex3 = mesh_faces[:,2] 
    vertex1 = mesh_points[faceVertexIndex1]
    vertex2 = mesh_points[faceVertexIndex2]
    vertex3 = mesh_points[faceVertexIndex3]
    faceMidPt = (vertex1 + vertex2 + vertex3)/3.0
    ptRayOrigin = faceMidPt + (sunVecRevNp * tol)
    ptRayTarget = faceMidPt + (sunVecRevNp * rayLength)

    return ptRayOrigin, ptRayTarget
    
def post_process_f(seg_idxs, f_count):
    #Rearrange intersection results
    face_in_sun = np.ones(f_count, dtype=bool)
    for ray_index in seg_idxs:
        face_in_sun[ray_index] = False

    return face_in_sun    


def raytrace_v(meshes, sun_vec_rev):
    
    mesh_tri = meshes.city_mesh
    volume = meshes.volume
    
    [pt_ray_origin, pt_ray_target] = pre_process_v(mesh_tri, sun_vec_rev)
    
    [seg_idxs, intersections, is_backface] = volume.intersections(pt_ray_origin, pt_ray_target,)
    
    [face_shading, vertex_in_sun, face_in_sun] = post_process_v(mesh_tri, seg_idxs)
    
    #print("---- Vertex intersection results ----")
    print("Found nr of intersections: " + str(len(seg_idxs)))

    return face_shading, vertex_in_sun, face_in_sun

def pre_process_v(mesh_tri, sun_vec_rev):
    mesh_points = mesh_tri.vertices    
    pt_ray_origin = np.zeros([len(mesh_points), 3])
    pt_ray_target = np.zeros([len(mesh_points), 3])
    tol = 0.01
    ray_length = 1000

    sunVecRevNp = np.array(sun_vec_rev) #Already numpy array?
    pt_ray_origin = mesh_points + (sunVecRevNp * tol)
    pt_ray_target = mesh_points + (sunVecRevNp * ray_length)
    
    return pt_ray_origin, pt_ray_target       

def post_process_v(meshTri, seg_idxs):
    vertex_in_sun = np.ones(len(meshTri.vertices), dtype = bool)
    for v_index_in_shade in seg_idxs:
        vertex_in_sun[v_index_in_shade] = False

    f_counter = 0
    face_shading = np.zeros(len(meshTri.faces), dtype=int)
    face_in_sun = np.zeros(len(meshTri.faces), dtype=bool)    
    for face in meshTri.faces: 
        v_counter = 0
        for v_index in face:
            if vertex_in_sun[v_index]:
                v_counter += 1
        face_shading[f_counter] = v_counter
        
        if(v_counter == 3):
            face_in_sun[f_counter] = True
        
        f_counter += 1

    return face_shading, vertex_in_sun, face_in_sun


def raytrace_skydome(volume:Volume, ray_targets, ray_areas):

    city_mesh_faces = volume.faces
    city_mesh_points = volume.points

    tol = 0.01
    ray_scale_factor = 1000             
    f_count = len(city_mesh_faces)  
    ray_count = len(ray_targets)

    faceVertexIndex1 = city_mesh_faces[:,0]
    faceVertexIndex2 = city_mesh_faces[:,1]
    faceVertexIndex3 = city_mesh_faces[:,2] 
    
    vertex1 = city_mesh_points[faceVertexIndex1]
    vertex2 = city_mesh_points[faceVertexIndex2]
    vertex3 = city_mesh_points[faceVertexIndex3]

    vector1 = vertex2 - vertex1
    vector2 = vertex3 - vertex1

    vector_cross = np.cross(vector1, vector2)
    vector_length = np.sqrt((vector_cross ** 2).sum(-1))[..., np.newaxis]
    normal = vector_cross / vector_length   

    face_mid_pt = (vertex1 + vertex2 + vertex3)/3.0
    pt_ray_origin = face_mid_pt + (normal * tol)
    ray_targets = ray_scale_factor * ray_targets
    sky_portion = np.zeros(f_count)

    for i in range(0, f_count):
        ray_o = np.array([pt_ray_origin[i,:]]) 
        ray_o_repeat = np.repeat(ray_o, ray_count, axis = 0)
        ray_t = ray_o_repeat + ray_targets
        [seg_idxs, intersections, is_backface] = volume.intersections(ray_o_repeat, ray_t)
        shaded_portion = np.sum(ray_areas[seg_idxs])
        sky_portion[i] = 1.0 - shaded_portion
        if((i % 100) == 0):
            print("Diffuse calculation for face:" + str(i) + " finished. Diffusion = " + str(sky_portion[i]))

    return sky_portion

def raytrace_skydome_debug(volume:Volume, ray_targets, face_indices):

    city_mesh_faces = volume.faces
    city_mesh_points = volume.points

    tol = 0.01
    ray_scale_factor = 1000.0    
    ray_count = len(ray_targets)
    all_seg_idxs = dict.fromkeys([i for i in range(0,len(face_indices))])
    all_face_mid_pts = np.zeros((len(face_indices),3))
    ray_targets_scaled = ray_scale_factor * ray_targets

    for i in range(0, len(face_indices)):
        index = face_indices[i]

        faceVertexIndex1 = city_mesh_faces[index,0]
        faceVertexIndex2 = city_mesh_faces[index,1]
        faceVertexIndex3 = city_mesh_faces[index,2] 
    
        vertex1 = city_mesh_points[faceVertexIndex1]
        vertex2 = city_mesh_points[faceVertexIndex2]
        vertex3 = city_mesh_points[faceVertexIndex3]

        vector1 = vertex2 - vertex1
        vector2 = vertex3 - vertex1

        vector_cross = np.cross(vector1, vector2)
        vector_length = np.sqrt((vector_cross ** 2).sum(-1))[..., np.newaxis]
        normal = vector_cross / vector_length   

        face_mid_pt = (vertex1 + vertex2 + vertex3)/3.0
        pt_ray_origin = face_mid_pt + (normal * tol)

        ray_o = np.array([pt_ray_origin]) 
        ray_o_repeat = np.repeat(ray_o, ray_count, axis = 0)
        ray_t = ray_o_repeat + ray_targets_scaled

        [seg_idxs, intersections, is_backface] = volume.intersections(ray_o_repeat, ray_t)

        all_seg_idxs[i] = seg_idxs
        all_face_mid_pts[i,:] = face_mid_pt

    return all_seg_idxs, all_face_mid_pts