import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os

import dtcc_solar.data_io as io 
from dtcc_solar.scripts.main import run_script
from dtcc_solar.sunpath import Sunpath, SunpathVis, SunpathUtils

from pprint import pp
from typing import List, Dict

from dtcc_solar import utils
from dtcc_solar.utils import Vec3
from dtcc_solar.sunpath import SunpathUtils
from typing import Dict, List
from shapely import LinearRing, Point
from csv import reader

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

        self.plot = True

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

        radius  = 1.0
        # Create sunpath
        sunpath = Sunpath(lat, lon, radius)
        [x, y, z, all_sun_pos] = sunpath.get_analemmas(2019, 5)

        # Import sunpath for the same location
        loop_pts = read_analemmas_from_csv_file(filename)
    
        # Prepare for comparison
        loop_pts = match_scale_of_imported_sunpath_diagram(loop_pts, radius)
        loops = create_sunpath_loops_as_linear_rings(loop_pts)

        if plot:
            ax = SunpathVis.initialise_plot(radius, name)
            SunpathVis.plot_analemmas(all_sun_pos, radius, ax, True, cmap, gmt_diff)
            SunpathVis.plot_imported_sunpath_diagarm(loop_pts, radius, ax, 'cool')

        # Distance between analemmas from trigonomety        
        dist_analemma = 2 * math.sin((math.pi /24)) * radius
    
        # Calculate the distance between the created and imported sunpath
        deviations = calc_distace_from_sun_pos_to_linear_rings(loops, all_sun_pos, gmt_diff)

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

 ######################## Utilitiy function used for test ##############################

def match_scale_of_imported_sunpath_diagram(loop_pts:Dict[int, List[Vec3]], radius: float) -> Dict[int, List[Vec3]]:
    # Calculate the correct scale factor for the imported sunpath diagram
    pt = loop_pts[0][0]
    current_raduis = math.sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z) 
    sf = radius / current_raduis

    for hour in loop_pts:
        for i in range(len(loop_pts[hour])):
            pt = loop_pts[hour][i]
            pt.x = sf * pt.x 
            pt.y = sf * pt.y
            pt.z = sf * pt.z
            loop_pts[hour][i] = pt

    return loop_pts

def create_sunpath_loops_as_linear_rings(loop_pts: Dict[int, List[Vec3]]) -> List[LinearRing]:

    loops = []
    for key in loop_pts:
        if(len(loop_pts[key]) > 0):
            coords = reformat_coordinates(loop_pts[key])
            linear_ring = LinearRing(coords)
            loops.append(linear_ring)
    
    return loops

def calc_distace_from_sun_pos_to_linear_rings(loops:List[LinearRing], all_sun_pos:Dict[int, List[Vec3]], gmt_diff: int) -> Dict[int, List[float]]:
    
    deviations = dict.fromkeys(range(0,24))

    for hour_utc in all_sun_pos:
        loop = loops[hour_utc]
        hour_local =  SunpathUtils.convert_local_time_to_utc(hour_utc, gmt_diff)    
        ds = []
        for i in range(len(all_sun_pos[hour_local])):
            sun_pos = all_sun_pos[hour_local][i]    
            pt = Point(sun_pos.x, sun_pos.y, sun_pos.z)
            d = loop.distance(pt)
            ds.append(d)

        deviations[hour_local] = ds    
        
    return deviations

def calc_distace_from_sun_to_rings(loops:List[LinearRing], sun_pos:Vec3) -> float:
    dist = np.zeros(len(loops))
    pt = Point(sun_pos.x, sun_pos.y, sun_pos.z)
    counter = 0
    for loop in loops:
        d = loop.distance(pt)
        dist[counter] = d
        counter += 1

    #Get distance to the closest loop
    min_dist = np.amin(dist)
    return min_dist

def get_distance_between_sun_loops(loop_pts:Dict[int, List[Vec3]], radius: float) -> List[float]:
    avrg_pts = []
    dist = np.zeros(24)
    dist_from_trigo = 2 * math.sin((math.pi /24)) * radius

    for key in loop_pts:
        z_max = -100000
        n = len(loop_pts[key])
        avrg_pt = Vec3(x = 0,y = 0,z = 0)
        for pt in loop_pts[key]:
            avrg_pt.x += (pt.x / n)
            avrg_pt.y += (pt.y / n)
            avrg_pt.z += (pt.z / n)

        avrg_pt = utils.normalise_vector3(avrg_pt)
        avrg_pt = utils.scale_vector3(avrg_pt, radius) 
        avrg_pts.append(avrg_pt)
    
    for i in range(len(avrg_pts)):
        pt = avrg_pts[i]
        pt_next = avrg_pts[0] 
        if(i < len(avrg_pts)-1):
            pt_next = avrg_pts[i+1]
        dx = pt.x - pt_next.x
        dy = pt.y - pt_next.y
        dz = pt.z - pt_next.z
        d = math.sqrt(dx*dx + dy*dy + dz*dz) 
        dist[i] = d

    return dist, dist_from_trigo

def reformat_coordinates(vec_list: List[Vec3]) -> np.ndarray:
    coords = np.zeros((len(vec_list), 3))
    for i in range(0, len(vec_list)):
        coords[i,0] = vec_list[i].x
        coords[i,1] = vec_list[i].y
        coords[i,2] = vec_list[i].z

    return coords

def read_analemmas_from_csv_file(filename:str):

    pts = []
    loop_pts = dict.fromkeys(range(0,24))
    loop_counter = 0

    with open(filename, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                if row[0] == '*':
                    loop_pts[loop_counter] = pts
                    loop_counter +=1
                    pts = []
                else:
                    # the row variable is a list that represents a row in csv
                    pt = Vec3(x = float(row[0]), y = float(row[1]), z = float(row[2]))                
                    pts.append(pt)

    loop_pts = remove_empty_dict_branch(loop_pts)

    return loop_pts

def remove_empty_dict_branch(a_dict: Dict[int, List[Vec3]]) -> Dict[int, List[Vec3]]:
    
    del_keys = []
    for key in a_dict:
        if not a_dict[key]:
            del_keys.append(key)

    for key in del_keys:
        del a_dict[key]

    return a_dict

#########################################################################################

if __name__ == "__main__":

    os.system('clear')

    print("--------------------- Sunpath test started -----------------------")

    test = TestSunpath()
    test.setup_method()
    test.test_sunpaths()



