import copy
import trimesh
import numpy as np

import dtcc_solar.raycasting as ray
import dtcc_solar.mesh_compute as mc
from dtcc_solar import utils
from trimesh import Trimesh


class Shading:
    def __init__(self, meshes):
        self.meshes = meshes
        self.mesh_collection = []
        self.faceShadingDict = {}
        self.faceAnglesDict = {}
        self.vertex_angles_dict = {}
        self.face_colors_dict = {}
        self.flux = 1000  # Watts per m2

        self.face_in_sun = 0
        self.face_sun_angles = 0
        self.face_irradiance = 0
        self.face_in_sun_sum = 0

    def execute_shading_v(self, sun_vec):
        iterCount = 10
        maxIter = 5
        edgeLengthAverage = mc.calculate_average_edge_length(self.meshes.city_mesh)
        targetLength = 0.05 * edgeLengthAverage
        evalMesh = copy.deepcopy(self.meshes.city_mesh)
        sun_vec_rev = utils.reverse_vector(sun_vec)
        all_face_in_sun = np.array([], dtype=bool)
        all_face_shading = np.array([])

        for i in range(0, iterCount):
            maxEdgeLength = (1 - i / iterCount) * (
                edgeLengthAverage - targetLength
            ) + targetLength
            [faceShading, vertexInSun, face_in_sun] = self.raytrace_v(
                evalMesh, self.meshes.volume, sun_vec_rev
            )
            borderFaceMask = mc.find_shadow_border_faces_rayV(evalMesh, faceShading)
            [
                meshNormal,
                meshBorder,
                face_shading_normal,
                face_in_sun_normal,
            ] = mc.split_mesh(evalMesh, borderFaceMask, faceShading, face_in_sun)
            all_face_in_sun = np.append(all_face_in_sun, face_in_sun_normal)
            mesh_border_SD = self.subdivide_border(meshBorder, maxEdgeLength, maxIter)
            evalMesh = mesh_border_SD  # Set the subDee mesh for next iteration!
            self.mesh_collection.append(meshNormal)
            all_face_shading = np.append(all_face_shading, face_shading_normal.tolist())

        # Calcuate the face shading for the finest Subdee level of the triangles
        [faceShadingSD, vertexInSun, face_in_sun] = ray.raytrace_v(
            mesh_border_SD, self.meshes.volume, sun_vec_rev
        )
        all_face_in_sun = np.append(all_face_in_sun, face_in_sun)
        self.mesh_collection.append(
            mesh_border_SD
        )  # Add the subDee mesh as the last step!
        all_face_shading = np.append(all_face_shading, faceShadingSD.tolist())

        joined_mesh = trimesh.util.concatenate(self.mesh_collection)

        [all_face_sun_angles, vertex_angles] = mc.face_sun_angle(joined_mesh, sun_vec)
        all_face_irradiance = mc.compute_irradiance(
            all_face_in_sun, all_face_sun_angles, self.flux, len(joined_mesh.faces)
        )

        self.face_in_sun = all_face_in_sun
        self.face_sun_angles = all_face_sun_angles
        self.face_irradiance = all_face_irradiance

        self.meshes.set_city_mesh_out(joined_mesh)

    def raytrace_v(self, meshes, sun_vec_rev):
        mesh_tri = meshes.city_mesh
        volume = meshes.volume

        [pt_ray_origin, pt_ray_target] = self._pre_process_v(mesh_tri, sun_vec_rev)

        [seg_idxs, intersections, is_backface] = volume.intersections(
            pt_ray_origin,
            pt_ray_target,
        )

        [face_shading, vertex_in_sun, face_in_sun] = self._post_process_v(
            mesh_tri, seg_idxs
        )

        # print("---- Vertex intersection results ----")
        print("Found nr of intersections: " + str(len(seg_idxs)))

        return face_shading, vertex_in_sun, face_in_sun

    def _pre_process_v(self, mesh_tri, sun_vec_rev):
        mesh_points = mesh_tri.vertices
        pt_ray_origin = np.zeros([len(mesh_points), 3])
        pt_ray_target = np.zeros([len(mesh_points), 3])
        tol = 0.01
        ray_length = 1000

        sunVecRevNp = np.array(sun_vec_rev)  # Already numpy array?
        pt_ray_origin = mesh_points + (sunVecRevNp * tol)
        pt_ray_target = mesh_points + (sunVecRevNp * ray_length)

        return pt_ray_origin, pt_ray_target

    def _post_process_v(self, meshTri, seg_idxs):
        vertex_in_sun = np.ones(len(meshTri.vertices), dtype=bool)
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

            if v_counter == 3:
                face_in_sun[f_counter] = True

            f_counter += 1

        return face_shading, vertex_in_sun, face_in_sun

    def find_shadow_border_faces_rayV(mesh, faceShading):
        borderFaceMask = np.ones(len(mesh.faces), dtype=bool)
        faces = list(mesh.faces)
        for i in range(len(faces)):
            if faceShading[i] < 3 and faceShading[i] > 0:
                borderFaceMask[i] = False

        return borderFaceMask

    def split_mesh(self, mesh: Trimesh, borderFaceMask, faceShading, face_in_sun):
        # Reversed face mask booleans
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

    def subdivide_border(self, meshBorder, maxEdgeLength, maxIter):
        [vs, fs] = trimesh.remesh.subdivide_to_size(
            meshBorder.vertices,
            meshBorder.faces,
            max_edge=maxEdgeLength,
            max_iter=maxIter,
            return_index=False,
        )
        meshBorderSD = trimesh.Trimesh(vs, fs)
        return meshBorderSD

    def calculate_average_edge_length(self, mesh):
        edges = mesh.edges_unique
        eCount = len(edges)
        vertices = list(mesh.vertices)
        edgeL = 0

        for edge in edges:
            vIndex1 = edge[0]
            vIndex2 = edge[1]
            d = utils.distance(vertices[vIndex1], vertices[vIndex2])
            edgeL += d

        edgeL = edgeL / eCount

        return edgeL
