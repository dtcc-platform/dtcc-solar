import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os

from dtcc_solar.scripts.main import run_script
from dtcc_solar import sun_utils
from dtcc_solar.sunpath import Sunpath

from pprint import pp
from typing import List, Dict
from dtcc_solar.utils import Vec3


class TestSunpath:

    locations: Dict
    plot: bool

    def setup_method(self):
        
        # Locations with predefined sunpath diagrams that have been generated using ladybug and the same settings as specified below.
        self.locations = {
            "Base"       :{ "name": "Base",           "lat":  51.5,  "lon": -0.12,   "tz": 'UTC',              "GMT_diff":  0, "cmap": 'autumn_r', "file": 'data/sunpaths/london.csv'           },
            "London"     :{ "name": "London",         "lat":  51.5,  "lon": -0.12,   "tz": 'Europe/London',    "GMT_diff":  0, "cmap": 'autumn_r', "file": 'data/sunpaths/london.csv'           },
            "Gothenburg" :{ "name": "Gothenburg",     "lat":  57.71, "lon":  11.97,  "tz": 'Europe/Stockholm', "GMT_diff":  1, "cmap": 'autumn_r', "file": 'data/sunpaths/gothenburg.csv'       },
            "Stockholm"  :{ "name": "Stockholm",      "lat":  59.33, "lon":  18.06,  "tz": 'Europe/Stockholm', "GMT_diff":  1, "cmap": 'autumn_r', "file": 'data/sunpaths/stockholm.csv'        },
            "Oslo"       :{ "name": "Oslo",           "lat":  59.91, "lon":  10.75,  "tz": 'Europe/Oslo',      "GMT_diff":  1, "cmap": 'autumn_r', "file": 'data/sunpaths/oslo.csv'             },
            "Helsinki"   :{ "name": "Helsinki",       "lat":  60.19, "lon":  24.94,  "tz": 'Europe/Helsinki',  "GMT_diff":  2, "cmap": 'autumn_r', "file": 'data/sunpaths/helsinki.csv'         },
            "Kiruna"     :{ "name": "Kiruna",         "lat":  67.51, "lon":  20.13,  "tz": 'Europe/Stockholm', "GMT_diff":  1, "cmap": 'autumn_r', "file": 'data/sunpaths/kiruna.csv'           }, 
            "NYC"        :{ "name": "New York",       "lat":  43.00, "lon": -75.00,  "tz": 'America/New_York', "GMT_diff": -5, "cmap": 'autumn_r', "file": 'data/sunpaths/nyc.csv'              },
            "Istanbul"   :{ "name": "Istanbul",       "lat":  41.01, "lon":  28.97,  "tz": 'Asia/Istanbul',    "GMT_diff":  3, "cmap": 'autumn_r', "file": 'data/sunpaths/istanbul.csv'         },
            "Rio"        :{ "name": "Rio de Janeiro", "lat": -22.90, "lon": -43.17,  "tz": 'America/Sao_Paulo',"GMT_diff": -3, "cmap": 'autumn_r', "file": 'data/sunpaths/rio_de_janeiro.csv'   } 
        }

        self.plot = False

########################### Testig of instant analysis ##############################

    def test_sunpaths(self):
        self.calc_sunpath_deviation(self.locations["London"], self.plot)
        self.calc_sunpath_deviation(self.locations["Gothenburg"], self.plot)
        self.calc_sunpath_deviation(self.locations["Stockholm"], self.plot)
        self.calc_sunpath_deviation(self.locations["Oslo"], self.plot)
        self.calc_sunpath_deviation(self.locations["Helsinki"], self.plot)
        self.calc_sunpath_deviation(self.locations["Kiruna"], self.plot)
        self.calc_sunpath_deviation(self.locations["NYC"], self.plot)
        self.calc_sunpath_deviation(self.locations["Istanbul"], self.plot)
        self.calc_sunpath_deviation(self.locations["Rio"], self.plot)
        pass


    def calc_sunpath_deviation(self, location, plot:bool):

        name = location["name"]
        filename = location["file"]
        lon = location["lon"]
        lat = location["lat"]
        gmt_diff = location["GMT_diff"]
        cmap = location["cmap"]

        radius  = 1
        origin = np.array([0, 0, 0])

        # Create sunpath
        sunpath = Sunpath(lat, lon, radius, origin)
        [x, y, z, all_sun_pos] = sunpath.get_analemmas(2019, 5)

        # Import sunpath for the same location
        loop_pts = sun_utils.read_sunpath_diagram_loops_from_csv_file(filename)
    
        # Prepare for comparison
        loop_pts = sun_utils.match_scale_of_imported_sunpath_diagram(loop_pts, radius)
        loops = sun_utils.create_sunpath_loops_as_linear_rings(loop_pts)

        if plot:
            ax = sun_utils.initialise_plot(radius, name)
            sun_utils.plot_analemmas(all_sun_pos, radius, ax, True, cmap, gmt_diff)
            sun_utils.plot_imported_sunpath_diagarm(loop_pts, radius, ax, 'cool')

        # Distance between analemmas from trigonomety        
        dist_analemma = 2 * math.sin((math.pi /24)) * radius
    
        # Calculate the distance between the created and imported sunpath
        deviations = sun_utils.calc_distace_from_sun_pos_to_linear_rings(loops, all_sun_pos, gmt_diff)

        error = []

        for hour in deviations:        
            max_dev = np.max(deviations[hour])
            avrg_dev = np.average(deviations[hour])
            relative_error_analemma = max_dev / dist_analemma
        
            # Comparing the deviation from the imported sun path with the radius of the sunpath diagram.
            relative_error_radius = max_dev / radius

            error.append(relative_error_radius)
            #Allowing 5 % diference between a solar position and the sunpath loop.
            if(relative_error_radius > 0.01 or relative_error_analemma > 0.04):
                print("Generated sunpath diagram does not match the imported sun paths which have been created using Ladybug in Grasshopper.")
                return False

        print("Deviation between generated sunpath and imported sunpath relative to the sun path radius: " + str(round(np.average(error) * 100, 4)) + "% " + "for the location " + name + ".")

        if plot:
            plt.show()

        return True


if __name__ == "__main__":

    os.system('clear')
    test = TestSunpath()
    test.setup_method()
    test.test_sunpaths()



