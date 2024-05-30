import numpy as np
import pprint
import copy
from dtcc_io import meshes
from dtcc_solar import py_embree_solar
from dtcc_model import Mesh, PointCloud
from pprint import pp
from dtcc_viewer import Scene, Window
from dtcc_solar.utils import calc_face_mid_points, calc_face_incircle


class TestEmbreeSolar:
    lat: float
    lon: float

    def setup_method(self):
        self.file_name = "../data/models/CitySurfaceS.stl"
        self.mesh = meshes.load_mesh(self.file_name)
        self.embree = py_embree_solar.PyEmbreeSolar(self.mesh.vertices, self.mesh.faces)

    def test_skydome(self):
        faces = self.embree.get_skydome_faces()
        vertices = self.embree.get_skydome_vertices()
        skydome_rays = self.embree.get_skydome_rays()

        pc = PointCloud(points=skydome_rays)
        mesh = Mesh(vertices=vertices, faces=faces)

        window = Window(1200, 800)
        scene = Scene()
        scene.add_pointcloud("Skydome rays", pc, size=0.01)
        scene.add_mesh("Skydome", mesh)
        window.render(scene)

    def test_sky_raytrace(self):
        success = self.embree.sky_raytrace_occ8()

        skyhit = self.embree.get_face_skyhit_results()
        svf = self.embree.get_sky_view_factor_results()

        window = Window(1200, 800)
        scene = Scene()
        scene.add_mesh("Mesh diffuse", self.mesh, data=svf)
        window.render(scene)

    def test_sky_raytrace2(self):
        success = self.embree.sky_raytrace_occ8()
        skyhit = self.embree.get_face_skyhit_results()
        svf = self.embree.get_sky_view_factor_results()
        skydome_rays = self.embree.get_skydome_rays()

        all_skydome_rays = []
        all_skydome_data = []
        face_centers, center_radia = calc_face_incircle(self.mesh)

        for i, face_center in enumerate(face_centers):
            dome_rays_copy = copy.deepcopy(skydome_rays)
            dome_rays_copy *= center_radia[i]
            dome_rays_copy += face_center
            all_skydome_rays.extend(dome_rays_copy)
            all_skydome_data.extend(skyhit[i])

        all_skydome_rays = np.array(all_skydome_rays)
        all_skydome_data = np.array(all_skydome_data)

        rays_pcs = PointCloud(points=all_skydome_rays)

        # window = Window(1200, 800)
        # scene = Scene()
        # scene.add_pointcloud("Skydome rays", rays_pcs, size=0.07, data=all_skydome_data)
        # scene.add_mesh("Mesh diffuse", self.mesh, data=skyprt)
        # window.render(scene)


if __name__ == "__main__":
    test = TestEmbreeSolar()
    test.setup_method()
    test.test_sky_raytrace2()
