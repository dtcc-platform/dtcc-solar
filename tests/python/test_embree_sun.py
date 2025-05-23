import numpy as np
import pprint
from dtcc_core.io import meshes
from dtcc_solar import py_embree_solar
from dtcc_core.model import PointCloud
from pprint import pp
from dtcc_viewer import Scene, Window


class TestEmbreeSolar:
    lat: float
    lon: float

    def setup_method(self):
        self.file_name = "../data/models/CitySurfaceS.stl"
        self.mesh = meshes.load_mesh(self.file_name)

        n_suns = 1000
        r = 500
        rot_step = np.linspace(0, 1 * np.pi, n_suns)
        x = r * np.sin(rot_step)
        y = r * np.cos(rot_step)
        z = r * abs(np.sin(rot_step / 1.0))
        self.sun_vecs = np.c_[x, y, z]

        self.embree = py_embree_solar.PyEmbreeSolar(self.mesh.vertices, self.mesh.faces)

    def test_skydome_raytrace(self):
        results1 = self.embree.sun_raytrace_occ1(self.sun_vecs)
        results3 = self.embree.sun_raytrace_occ8(self.sun_vecs)

        angles = self.embree.get_angle_results()
        occlusion = self.embree.get_occluded_results()

        angles = np.sum(-1 * angles, axis=0)
        occlusion = np.sum(-1 * occlusion, axis=0)

        pc = PointCloud(points=self.sun_vecs)

        window = Window(1200, 800)
        scene = Scene()
        scene.add_pointcloud("pc", pc, size=1)
        scene.add_mesh("Mesh anlges", self.mesh, data=angles)
        scene.add_mesh("Mesh occlusion", self.mesh, data=occlusion)
        window.render(scene)


if __name__ == "__main__":
    test = TestEmbreeSolar()
    test.setup_method()
    # test.test_skydome_raytrace()
    test.test_skydome_raytrace()
