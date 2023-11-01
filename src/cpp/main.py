import sys
sys.path.insert(0, './build')
import py_embree_solar
import numpy as np 

if __name__ == "__main__":
    # create raytracer instance
    embree_solar = py_embree_solar.PyEmbreeSolar()

    # Sun vector 1
    sun_vec_1 =  np.array([0.0, 0.0, 1.0])
    sun_vec_2 =  np.array([0.0, 1.0, 1.0])
    sun_vec_3 =  np.array([1.0, 1.0, 1.0])
    
    # All sun vectors
    sun_vecs = np.array([sun_vec_1, sun_vec_2, sun_vec_3])

    # run sun analysis
    #embree_solar.sun_raytrace_occ1(sun_vecs)
    #embree_solar.sun_raytrace_occ4(sun_vecs)
    #embree_solar.sun_raytrace_occ8(sun_vecs)
    #embree_solar.sun_raytrace_occ16(sun_vecs)

    # run sky analysis
    embree_solar.sky_raytrace_occ1()
    embree_solar.sky_raytrace_occ4()
    embree_solar.sky_raytrace_occ8()
    embree_solar.sky_raytrace_occ16()






