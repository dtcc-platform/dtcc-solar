import numpy as np
from dtcc_solar import py_embree_solar
from dtcc_io import meshes
from pprint import pp


class TestEmbreeSolar:
    lat: float
    lon: float

    def setup_method(self):
        self.file_name = "../data/models/CitySurfaceS.stl"
        self.mesh = meshes.load_mesh(self.file_name)

        # Sun vector 1
        sun_vec_1 = np.array([0.0, 0.0, 1.0])
        sun_vec_2 = np.array([0.0, 1.0, 1.0])
        sun_vec_3 = np.array([1.0, 1.0, 1.0])

        # All sun vectors
        self.sun_vecs = np.array([sun_vec_1, sun_vec_2, sun_vec_3])

        self.es = py_embree_solar.PyEmbreeSolar()

        self.faces = self.es.getMeshFaces()

    def test_sun_raytrace(self):
        results1 = self.es.sun_raytrace_occ1(self.sun_vecs)
        results2 = self.es.sun_raytrace_occ4(self.sun_vecs)
        results3 = self.es.sun_raytrace_occ8(self.sun_vecs)
        results4 = self.es.sun_raytrace_occ16(self.sun_vecs)

        assert len(results1[0]) == len(self.faces)

    def test_sky_raytrace(self):
        # run sky analysis
        results1 = self.es.sky_raytrace_occ1()
        results2 = self.es.sky_raytrace_occ4()
        results3 = self.es.sky_raytrace_occ8()
        results4 = self.es.sky_raytrace_occ16()

        assert len(results1) == len(self.faces)
