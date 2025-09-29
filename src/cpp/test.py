import sys
sys.path.insert(0, './build')
import py_embree_solar
import time

if __name__ == "__main__":
    # create raytracer instance
    embree_solar = py_embree_solar.PyEmbreeSolar()

    # run analysis
    start_time = time.time()
    
    for i in range(1000):
        print(i)
        embree_solar.raytrace_occ8()
        

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    