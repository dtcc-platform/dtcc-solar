import numpy as np
import pandas as pd
import sys
import pathlib
import csv
import math
from csv import reader
project_dir = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(project_dir)

from dtcc_solar.utils import AnalysisType, ColorBy, Vec3
from typing import Dict, List
import typing
from pprint import pp

from shapely import LinearRing, Point, get_z

class Parameters:
    def __init__(self, a_type, fileName, lat, lon, prep_disp, disp, origin, color_by, export, one_date, s_date, e_date):
        self.a_type = AnalysisType(a_type)
        self.file_name = fileName
        self.latitude = lat
        self.longitude = lon 
        self.prepare_display = bool(prep_disp)
        self.display = bool(disp)
        self.origin = np.array(origin, dtype= float)
        self.color_by = ColorBy(color_by)
        self.export = export
        self.one_date = one_date
        self.start_date = s_date
        self.end_date = e_date
        self.discretisation = '1H' 

def export_results(solpos):
    with open("sunpath.txt", "w") as f:
        
        for item in solpos['zenith'].values:
            f.write(str(item[0]) + '\n')

def print_list(listToPrint, path):
    counter = 0
    with open(path, 'w') as f:
        for row in listToPrint:
            f.write( str(row) + '\n')
            counter += 1 

    print("Export completed")

def print_dict(dictToPrint, filename):
    counter = 0
    with open(filename, "w") as f:
        for key in dictToPrint:
            f.write('Key:' + str(key) + ' ' + str(dictToPrint[key])+'\n')


def print_results(shouldPrint,faceRayFaces):
    counter = 0
    if shouldPrint:
        with open("faceRayFace.txt", "w") as f:
            for key in faceRayFaces:
                f.write('Face index:' + str(key) + ' ' + str(faceRayFaces[key])+'\n')
                counter += 1 

    print(counter)

def import_weather_date_epw(p:Parameters):
    file = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw' 
    year_dict_keys = []
    direct_normal_radiation = []
    diffuse_horisontal_radiation = []
    year_weather_data = dict()
    search = False

    with open(file, 'r') as f:
        for line in f:
            if search:
                line_segs = line.split(',')
                year = line_segs[0]
                month = make_double_digit_str(line_segs[1])
                day = make_double_digit_str(line_segs[2])
                hour = make_double_digit_str(line_segs[3])
                date_key = year + '-' + month + '-' + day + ' ' + hour + ':00:00'
                year_dict_keys.append(date_key)
                direct_normal_radiation.append(float(line_segs[14]))  
                diffuse_horisontal_radiation.append(float(line_segs[15]))
                
            #The first 9 lines contains information of the data structure
            if line[0:12] == 'DATA PERIODS':
                search = True    

    for i, key in enumerate(year_dict_keys):
        d_type = ['diffuse_radiation', 'direct_normal_radiation']
        a_dict = dict.fromkeys([d for d in d_type])
        a_dict['diffuse_radiation'] = diffuse_horisontal_radiation[i]
        a_dict['direct_normal_radiation'] = direct_normal_radiation[i] 
        year_weather_data[key] = a_dict
        
    print_dict(year_weather_data, '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/output/w_data_year_epw.txt')

def make_double_digit_str(s:str):
    if len(s) == 1:
        s = '0' + s
    return s     

def find_date_range_epw(file):
    start_year = 2200
    end_year = 1800
    search = False
    with open(file, 'r') as f:
        for line in f:
            if search:
                year = int(line[0:4]) 
                if year < start_year:
                    start_year = year
                if year > end_year:
                    end_year = year
                    
            #The first 9 lines contains information of the data structure
            if line[0:12] == 'DATA PERIODS':
                search = True       

    start_date = str(start_year) + '-01-01 00:00:00'
    end_date = str(end_year) + '-12-31 23:59:00'

    return start_date, end_date

def import_weather_data_clm(p:Parameters):
    
    line_index = 0
    data_counter = 0
    #The data file contains weather data for each hour during the year 2015.
    file = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm'
    year_dates = pd.date_range(start = '2015-01-01 00:00:00', end = '2015-12-31 23:59:00', freq = '1H')
    year_dict_keys = np.array([str(d) for d in year_dates])
    year_weather_data = dict.fromkeys(year_dict_keys)

    with open(file, 'r') as f:
        for line in f:
            #The first 12 lines contains information of the data structure
            if line_index > 12 and line[0] != '*':
               year_weather_data[str(year_dates[data_counter])] = line2numbers(line)     
               data_counter += 1     
            
            line_index += 1

    print_dict(year_weather_data, '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/output/w_data_year_clm.txt')
    
    if( p.a_type == AnalysisType.sun_raycasting or 
        p.a_type == AnalysisType.sky_raycasting or
        p.a_type == AnalysisType.sky_raycasting_some):
        sub_dates = pd.date_range(start = p.one_date, end = p.one_date, freq = '1H')
    else:
        sub_dates = pd.date_range(start = p.start_date, end = p.end_date, freq = '1H')

    [sub_weather_data, sub_dict_keys] = get_weather_data_subset(year_weather_data, sub_dates)    

    print_dict(sub_weather_data, '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/output/w_data_sub.txt')

    return sub_weather_data, sub_dict_keys, sub_dates

def get_weather_data_subset(year_weather_data, sub_dates):
    sub_dict_keys = np.array([str(d) for d in sub_dates])
    sub_weather_data = dict.fromkeys(sub_dict_keys)

    for key in sub_dict_keys:
        sub_weather_data[key] = year_weather_data[key]

    return sub_weather_data, sub_dict_keys

def line2numbers(line):
    keys = ['diffuse_radiation', 'direct_normal_radiation']
    adict = dict.fromkeys([k for k in keys])
    line = line.replace('\n', '')
    linesegs = line.split(',')
    if(len(linesegs) == 6):
        adict['diffuse_radiation'] = float(linesegs[0])
        adict['direct_normal_radiation'] = float(linesegs[2])

    return adict

def WriteToCsv(filename:str, data):

    print("Write to CSV")
    print(type(data))

    data_list = data.to_list()

    print(type(data_list))

    with open(filename, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        for d in data:
            filewriter.writerow(str(d))
        
def read_sunpath_diagram_from_csv_file(filename):

    pts = []
    with open(filename, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                pt = Vec3(x = float(row[0]), y = float(row[1]), z = float(row[2]))                
                pts.append(pt)

    return pts

def sunpath_testing(filename:str, all_sun_pos: Dict[int, List[Vec3]], radius:float):

    print("Sunpath testing")
    loop_pts = read_sunpath_diagram_loops_from_csv_file(filename)

    loop_pts = match_scale_of_imported_sunpath_diagram(loop_pts, radius)
    
    loops = create_sunpath_loops_as_linear_rings(loop_pts)

    loops_dist = get_distance_between_sun_loops(loop_pts)

    print(loops_dist)

    deviations = calc_distace_from_sun_pos_to_linear_rings(loops, all_sun_pos)

    avrg_loops_dist = np.average(loops_dist)

    print("Average distance between sunpath loops is:" + str(avrg_loops_dist))

    print("The deviation between sun position and sunpath loops is:")
    
    for hour in deviations:        
        max_dev = np.max(deviations[hour])
        avrg_dev = np.average(deviations[hour])
        print("Max deviation: " + str(max_dev) + "Avrg deviation: " + str(avrg_dev))
        relative_error = max_dev / avrg_loops_dist
        
        #Allowing 5 % diference between a solar position and the sunpath loop.
        if(relative_error > 0.05):
            return False

        print("Error relative to the loop distance: " + str(relative_error))

    return True

def read_sunpath_diagram_loops_from_csv_file(filename:str):

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

def match_scale_of_imported_sunpath_diagram(loop_pts:Dict[int, List[Vec3]], radius: float) -> Dict[int, List[Vec3]]:
    # Calculate the correct scale factor for the imported sunpath diagram
    pt = loop_pts[0][0]
    current_raduis = math.sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z) 
    sf = radius / current_raduis

    for hour in loop_pts:
        print(len(loop_pts[hour]))
        for i in range(len(loop_pts[hour])):
            pt = loop_pts[hour][i]
            pt.x = sf * pt.x 
            pt.y = sf * pt.y
            pt.z = sf * pt.z
            loop_pts[hour][i] = pt

    return loop_pts

def remove_empty_dict_branch(a_dict: Dict[int, List[Vec3]]) -> Dict[int, List[Vec3]]:
    
    del_keys = []
    for key in a_dict:
        if not a_dict[key]:
            del_keys.append(key)

    for key in del_keys:
        del a_dict[key]

    return a_dict

def create_sunpath_loops_as_linear_rings(loop_pts: Dict[int, List[Vec3]]) -> List[LinearRing]:

    loops = []
    for key in loop_pts:
        if(len(loop_pts[key]) > 0):
            coords = reformat_coordinates(loop_pts[key])
            linear_ring = LinearRing(coords)
            loops.append(linear_ring)
    
    return loops

def calc_distace_from_sun_pos_to_linear_rings(loops:List[LinearRing], all_sun_pos:Dict[int, List[Vec3]]) -> Dict[int, List[float]]:
    
    deviations = dict.fromkeys(range(0,24))

    for hour in all_sun_pos:
        d_min = 100000000
        loop = loops[hour]    
        ds = []
        for i in range(len(all_sun_pos[hour])):
            sun_pos = all_sun_pos[hour][i]    
            pt = Point(sun_pos.x, sun_pos.y, sun_pos.z)
            d = loop.distance(pt)
            ds.append(d)

        deviations[hour] = ds    
        
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

def get_distance_between_sun_loops(loop_pts:Dict[int, List[Vec3]]) -> List[float]:
    avrg_pts = []
    dist = np.zeros(24)
    n_loops = len(loop_pts)
    for key in loop_pts:
        z_max = -100000
        n = len(loop_pts[key])
        avrg_pt = Vec3(x=0,y=0,z=0)
        for pt in loop_pts[key]:
            avrg_pt.x += (pt.x / n)
            avrg_pt.y += (pt.y / n)
            avrg_pt.z += (pt.z / n)

        avrg_pts.append(avrg_pt)
        print(avrg_pt)
    
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

    return dist

def reformat_coordinates(vec_list: List[Vec3]) -> np.ndarray:

    coords = np.zeros((len(vec_list), 3))
    for i in range(0, len(vec_list)):
        coords[i,0] = vec_list[i].x
        coords[i,1] = vec_list[i].y
        coords[i,2] = vec_list[i].z

    return coords

