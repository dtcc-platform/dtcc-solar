import sys
sys.path.insert(0, './build')
import py_embree_solar

if __name__ == "__main__":
    # create raytracer instance
    embree_solar = py_embree_solar.PyEmbreeSolar()

    # run analysis
    embree_solar.raytrace_occ1()
    embree_solar.raytrace_occ4()
    embree_solar.raytrace_occ8()
    embree_solar.raytrace_occ16()






