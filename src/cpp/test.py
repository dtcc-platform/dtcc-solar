import sys
sys.path.insert(0, './build')
import py_solar
import time

if __name__ == "__main__":
    # create raytracer instance
    solar = py_solar.PySolar()

    # run analysis
    start_time = time.time()

    # Note: raytrace_occ8 has been removed - use run_2_phase_analysis or run_3_phase_analysis
    # This test script needs to be updated with proper test data

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")
