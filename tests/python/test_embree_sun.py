import numpy as np
from dtcc_io import meshes
from dtcc_solar import py_embree_solar
from dtcc_model import Mesh, PointCloud
from pprint import pp
from dtcc_viewer import Scene, Window, MeshShading


class TestEmbreeSolar:
    lat: float
    lon: float

    def setup_method(self):
        self.file_name = "../data/models/CitySurfaceS.stl"
        self.mesh = meshes.load_mesh(self.file_name)

        n_suns = 100
        r = 50
        rot_step = np.linspace(0, 1 * np.pi, n_suns)
        x = r * np.sin(rot_step)
        y = r * np.cos(rot_step)
        z = r * abs(np.sin(rot_step / 1.0))
        self.sun_vecs = np.c_[x, y, z]

        self.es = py_embree_solar.PyEmbreeSolar(self.mesh.vertices, self.mesh.faces)

    def test_skydome(self):
        faces = self.es.getSkydomeFaces()
        vertices = self.es.getSkydomeVertices()
        skydome_rays = self.es.getSkydomeRays()

        pc = PointCloud(points=skydome_rays)
        mesh = Mesh(vertices=vertices, faces=faces)

        window = Window(1200, 800)
        scene = Scene()
        scene.add_pointcloud("Skydome rays", pc, size=0.01)
        scene.add_mesh("Skydome", mesh)
        window.render(scene)

    def test_skydome_raytrace(self):
        # run analysis

        results1 = self.es.sun_raytrace_occ1(self.sun_vecs)
        # results2 = self.es.sun_raytrace_occ4(self.sun_vecs)
        # results3 = self.es.sun_raytrace_occ8(self.sun_vecs)
        # results4 = self.es.sun_raytrace_occ16(self.sun_vecs)

        results = np.sum(-1 * results1, axis=0)

        pc = PointCloud(points=self.sun_vecs)

        window = Window(1200, 800)
        scene = Scene()
        scene.add_pointcloud("pc", pc, size=1)
        scene.add_mesh("Skydome", self.mesh, results)
        window.render(scene)
