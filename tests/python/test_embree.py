import numpy as np
from dtcc_solar import py_embree_solar as embree
from dtcc_core.io import meshes
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
        self.embree = embree.PyEmbreeSolar()
        self.faces = self.embree.get_mesh_faces()

    def test_sun_raytrace(self):
        success1 = self.embree.sun_raytrace_occ1(self.sun_vecs)
        success2 = self.embree.sun_raytrace_occ4(self.sun_vecs)
        success3 = self.embree.sun_raytrace_occ8(self.sun_vecs)
        success4 = self.embree.sun_raytrace_occ16(self.sun_vecs)

        assert success1 and success2 and success3 and success4

    def test_sky_raytrace(self):
        # run sky analysis
        success1 = self.embree.sky_raytrace_occ1()
        success2 = self.embree.sky_raytrace_occ4()
        success3 = self.embree.sky_raytrace_occ8()
        success4 = self.embree.sky_raytrace_occ16()

        assert success1 and success2 and success3 and success4
