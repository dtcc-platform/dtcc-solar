import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pvlib import solarposition
from dtcc_solar import utils
from pprint import pp
from dtcc_solar.data_io import Vec3
import pytz
from timezonefinder import TimezoneFinder
import datetime

from dtcc_solar.utils import Vec3
from typing import Dict, List
from pprint import pp
from shapely import LinearRing, Point, get_z
from csv import reader
from dtcc_solar import utils


def convert_utc_to_local_time(utc_h, gmt_diff):

    local_h = utc_h + gmt_diff
    if(local_h < 0):
        local_h = 24 + local_h
    elif (local_h > 23):    
        local_h = local_h - 24

    return local_h

def convert_local_time_to_utc(local_h, gmt_diff):

    utc_h = local_h - gmt_diff
    if(utc_h < 0):
        utc_h = 24 + utc_h
    elif (utc_h > 23):    
        utc_h = utc_h - 24

    return utc_h

def get_timezone_from_long_lat(lat, long):
    tf = TimezoneFinder() 
    timezone_str = tf.timezone_at(lng=long, lat=lat)
    print(timezone_str)

    timezone = pytz.timezone(timezone_str)
    dt = datetime.datetime.now()
    offset = timezone.utcoffset(dt)
    h_offset_1 = offset.seconds / 3600 
    h_offset_2 = 24 - h_offset_1

    print("Time zone: " + str(timezone))
    print("GMT_diff: " + str(h_offset_1) + " or: " + str(h_offset_2))
    
    h_offset = np.min([h_offset_1, h_offset_2])

    return h_offset

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

def shift_sun_pos_dict_for_timezone(all_sun_pos:Dict[int, List[Vec3]], gmt_diff: float):
 
    pass


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

def calc_distace_from_sun_pos_to_linear_rings(loops:List[LinearRing], all_sun_pos:Dict[int, List[Vec3]], gmt_diff: int) -> Dict[int, List[float]]:
    
    deviations = dict.fromkeys(range(0,24))

    for hour_utc in all_sun_pos:
        loop = loops[hour_utc]
        hour_local = convert_local_time_to_utc(hour_utc, gmt_diff)    
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

def initialise_plot(r, title):
    plt.rcParams['figure.figsize'] = (16,11)
    plt.title(label=title, fontsize=44, color="black")
    ax = plt.axes(projection = '3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    return ax

def plot_imported_sunpath_diagarm(pts, radius, ax, cmap):
    
    n = int(0.5 * len(pts[0]))
    x,y,z = [],[],[]
    for hour in pts:
        for i in range(n):
            pt = pts[hour][i]
            x.append(pt.x)
            y.append(pt.y)
            z.append(pt.z)    
    
    ax.scatter3D(x, y, z, c=z, cmap = cmap, vmin = 0, vmax = radius)


def plot_analemmas(all_sun_pos: Dict[int, List[Vec3]],radius, ax, plot_night, cmap, gmt_diff):
    
    x,y,z = [],[],[]
    z_max_indices = []
    counter = 0
    
    for hour in all_sun_pos:
        z_max = -100000000
        z_max_index = -1
        for i in range(len(all_sun_pos[hour])):
            x.append(all_sun_pos[hour][i].x) 
            y.append(all_sun_pos[hour][i].y) 
            z.append(all_sun_pos[hour][i].z)
            if(all_sun_pos[hour][i].z > z_max):
                z_max = all_sun_pos[hour][i].z    
                z_max_index = counter
            counter += 1
        z_max_indices.append(z_max_index)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    day_indices = np.where(z > 0)
    night_indices = np.where(z <= 0)

    ax.scatter3D(x[day_indices],y[day_indices],z[day_indices], c=z[day_indices] , cmap = cmap, vmin = 0, vmax = radius)
    if plot_night:
        ax.scatter3D(x[night_indices],y[night_indices],z[night_indices], color = 'w')

    # Add label to the hourly loops
    for i in range(len(z_max_indices)):
        utc_hour = i
        local_hour = convert_utc_to_local_time(utc_hour, gmt_diff)
        text_pos_max = np.array([x[z_max_indices[i]], y[z_max_indices[i]], z[z_max_indices[i]]])
        ax.text(text_pos_max[0], text_pos_max[1], text_pos_max[2], str(local_hour), fontsize=12)


def plot_daypath(x_dict, y_dict, z_dict, radius, ax, plot_night):

    for key in x_dict:
        x = x_dict[key]
        y = y_dict[key]
        z = z_dict[key]

        day_indices = np.where(z > 0)
        night_indices = np.where(z <= 0)
        z_color = z[day_indices]
        ax.scatter3D(x[day_indices],y[day_indices],z[day_indices], c=z_color , cmap = 'autumn_r', vmin = 0, vmax = radius)
        if plot_night:    
            ax.scatter3D(x[night_indices],y[night_indices],z[night_indices], color = 'w')

def plot_day_loop_with_text(x, y, z, h, radius, ax, plot_night, cmap):
    day_indices = np.where(z > 0)
    night_indices = np.where(z <= 0)
    ax.scatter3D(x[day_indices],y[day_indices],z[day_indices], c=z[day_indices] , cmap = cmap, vmin = 0, vmax = radius)
    if plot_night:
        ax.scatter3D(x[night_indices],y[night_indices],z[night_indices], color = 'w')
    max_z_indices = np.where(z == np.max(z))
    max_z_index = max_z_indices[0][0]
    text_pos_max = np.array([x[max_z_index], y[max_z_index], z[max_z_index]])
    ax.text(text_pos_max[0], text_pos_max[1], text_pos_max[2], str(h), fontsize=12)

    min_z_indices = np.where(z == np.min(z))
    min_z_index = min_z_indices[0][0]
    text_pos_min = np.array([x[min_z_index], y[min_z_index], z[min_z_index]])
    ax.text(text_pos_min[0], text_pos_min[1], text_pos_min[2], str(h), fontsize=12)


def plot_single_sun(x, y, z, radius, ax):
    z_color = z
    ax.scatter3D(x,y,z, c=z_color , cmap = 'autumn_r', vmin = 0, vmax = radius)        

def plot_multiple_suns(sun_pos, radius, ax, plot_night):
    z = sun_pos[:,2]
    day_indices = np.where(z > 0)
    night_indices = np.where(z <= 0)
    zColor = z[day_indices]
    ax.scatter3D(sun_pos[day_indices,0],sun_pos[day_indices,1],sun_pos[day_indices,2], c=zColor , cmap = 'autumn_r', vmin = 0, vmax = radius)
    if plot_night:    
        ax.scatter3D(sun_pos[night_indices,0],sun_pos[night_indices,1],sun_pos[night_indices,2], color = 'w')        
