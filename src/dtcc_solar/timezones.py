import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import sys
import pathlib
from pvlib import solarposition
from dtcc_solar import utils
from pprint import pp
import dtcc_solar.data_io
from dtcc_solar.data_io import Vec3
import pytz
from tzwhere import tzwhere
from timezonefinder import TimezoneFinder
import datetime
import shapely

def initialise_plot(r):
    plt.rcParams['figure.figsize'] = (16,11)
    ax = plt.axes(projection = '3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    return ax

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


def plot_imported_sunpath_diagarm(pts, radius, ax, cmap):
    
    current_raduis = math.sqrt(pts[0].x * pts[0].x + pts[0].y * pts[0].y + pts[0].z * pts[0].z) 
    sf = radius / current_raduis
    x,y,z = [],[],[]
    for pt in pts:
        x.append(sf * pt.x)
        y.append(sf * pt.y)
        z.append(sf * pt.z)
    
    ax.scatter3D(x, y, z, c=z, cmap = cmap, vmin = 0, vmax = radius)
    
def get_one_day_loop(radius, location):

    ax = initialise_plot(radius)
    start_date = '2015-01-01 00:00:00'
    end_date = '2015-01-01 23:00:00'    
    
    times_no = pd.date_range(start = start_date, end = end_date, freq = 'H', tz = 'UTC')
    times_tz = pd.date_range(start = start_date, end = end_date, freq = 'H', tz = location["tz"])    

    print(type(times_no))
    print(times_no)
    print(type(times_tz))
    print(times_tz)

    sol_pos_no = solarposition.get_solarposition(times_no, location["lat"], location["lon"])
    sol_pos_tz = solarposition.get_solarposition(times_tz, location["lat"], location["lon"])

    print(sol_pos_no)
    print(sol_pos_tz)
    

def get_sunpath_hour_loops(year, sample_rate , plot_results, plot_night, radius, ax, location, name_extension, is_tz_dep):

    start_date = str(year) + '-01-01 00:00:00'
    end_date = str(year) + '-12-31 23:00:00'

    if is_tz_dep:    
        times = pd.date_range(start = start_date, end = end_date, freq = 'H', tz = location["tz"])
    else:
        times = pd.date_range(start = start_date, end = end_date, freq = 'H')

    # Times to evaluate will be 8760 for a normal year and 8764 for leep year  
    times_to_evaluate = len(times.values)
    print("Times to evaluate: " + str(times_to_evaluate))
        
    # Number of days will be 365 or 366 at leep year
    num_of_days = int(np.ceil(times_to_evaluate / 24))

    print("Number of days: " + str(num_of_days))
    # Get solar position from the pvLib libra
    sol_pos_hour = solarposition.get_solarposition(times, location["lat"], location["lon"])

    filename = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/output/debug' + name_extension + '.csv'
    dtcc_solar.data_io.print_list(sol_pos_hour.index.to_list(), filename)

    print("Size sol_pos_hour: " + str(len(sol_pos_hour)))

    # Reduce the number of days to reduce the density of the sunpath diagram
    days = np.zeros(num_of_days)
    num_evaluated_days = len(days[0:len(days):sample_rate])

    print("Number of evaluated days: " + str(num_evaluated_days))  

    mat_elev_hour = np.zeros((24, num_evaluated_days))
    mat_azim_hour = np.zeros((24, num_evaluated_days))
    mat_zeni_hour = np.zeros((24, num_evaluated_days))

    loop_hours = np.unique(sol_pos_hour.index.hour)

    my_Datetimeindex = sol_pos_hour.index

    if is_tz_dep:
        tz_offset = np.zeros(times_to_evaluate) 
        counter = 0
        for time in sol_pos_hour.index.timetz:
            offset = time.tzinfo.utcoffset(time)
            tz_offset[counter] = offset.seconds / 3600 
            counter += 1

    filename2 = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/output/time' + name_extension + '.csv'
    dtcc_solar.data_io.print_list(my_Datetimeindex.timetz, filename2)

    x = dict.fromkeys([h for h in loop_hours])
    y = dict.fromkeys([h for h in loop_hours])
    z = dict.fromkeys([h for h in loop_hours])
    sun_pos = dict.fromkeys([h for h in loop_hours])

    # Get hourly sun path loops in matrix form and elevaion, azimuth and zenith coordinates
    for hour in loop_hours:
        subset = sol_pos_hour.loc[sol_pos_hour.index.hour == hour, :]         
        #subset = utils.remove_date_range_duplicates(subset)  
         
        rad_elev = np.radians(subset.apparent_elevation)
        rad_azim = np.radians(subset.azimuth)
        rad_zeni = np.radians(subset.zenith)
        mat_elev_hour[hour, :] = rad_elev.values[0:len(rad_elev.values):sample_rate]
        mat_azim_hour[hour, :] = rad_azim.values[0:len(rad_azim.values):sample_rate]
        mat_zeni_hour[hour, :] = rad_zeni.values[0:len(rad_zeni.values):sample_rate]

    # Convert hourly sun path loops from spherical to cartesian coordiantes
    for h in range(0,24):
        x[h] = radius * np.cos(mat_elev_hour[h, :]) * np.cos(-mat_azim_hour[h, :])
        y[h] = radius * np.cos(mat_elev_hour[h, :]) * np.sin(-mat_azim_hour[h, :])
        z[h] = radius * np.sin(mat_elev_hour[h, :])
        local_h = convert_utc_to_local_time(h, location['GMT_diff'])
        utc_h = convert_local_time_to_utc(local_h, location['GMT_diff'])
        sun_pos[h] = utils.create_list_of_vectors(x[h], y[h], z[h])
        
        if plot_results:
            plot_day_loop_with_text(x[h], y[h], z[h], local_h, radius, ax, plot_night, location["cmap"])

    return sun_pos

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


if __name__ == "__main__":

    os.system('clear')
    print("--------------- My Solar Example Started -------------")
    origin = np.array([0, 0, 0])
    location = {
        "Base"       :{ "lat": 51.5,  "lon": -0.12,   "tz": 'UTC',              "GMT_diff":  0, "cmap": 'cool'     },
        "London"     :{ "lat": 51.5,  "lon": -0.12,   "tz": 'Europe/London',    "GMT_diff":  0, "cmap": 'autumn_r' },
        "Gothenburg" :{ "lat": 57.71, "lon": 11.97,   "tz": 'Europe/Stockholm', "GMT_diff":  1, "cmap": 'autumn_r' },
        "Stockholm"  :{ "lat": 59.33, "lon": 18.06,   "tz": 'Europe/Stockholm', "GMT_diff":  1, "cmap": 'cool'     },
        "Oslo"       :{ "lat": 59.91, "lon": 10.75,   "tz": 'Europe/Oslo',      "GMT_diff":  1, "cmap": 'autumn_r' },
        "Helsinki"   :{ "lat": 60.19, "lon": 24.94,   "tz": 'Europe/Helsinki',  "GMT_diff":  2, "cmap": 'autumn_r' },
        "Umeå"       :{ "lat": 63.82, "lon": 20.26,   "tz": 'Europe/Stockholm', "GMT_diff":  1, "cmap": 'autumn_r' },
        "Luleå"      :{ "lat": 65.58, "lon": 22.15,   "tz": 'Europe/Stockholm', "GMT_diff":  1, "cmap": 'autumn_r' },
        "Kiruna"     :{ "lat": 67.51, "lon": 20.13,   "tz": 'Europe/Stockholm', "GMT_diff":  1, "cmap": 'autumn_r' }, 
        "NYC"        :{ "lat": 43.00, "lon": -75.00,  "tz": 'America/New_York', "GMT_diff": -5, "cmap": 'autumn_r' }           
    }

    get_timezone_from_long_lat(location["NYC"]["lat"], location["NYC"]["lon"])

    radius = 1.0
    horizon_z = 0.0
    year = 2015
    name_1 = "1"
    name_2 = "2"
    ax = initialise_plot(radius)
    all_sun_pos = get_sunpath_hour_loops(year, 5, True, True, radius, ax, location["London"], name_1, False)
    
    print("-------------------------------------------")
    #get_sunpath_hour_loops(year, 5, True, True, radius, ax, location["NYC"], name_2, True)

    filename = '../../sandbox/sunpath_london.csv'
    pts = dtcc_solar.data_io.read_sunpath_diagram_from_csv_file(filename)
    plot_imported_sunpath_diagarm(pts, radius, ax, 'summer_r')

    filename2 = '../../sandbox/sunpath_london_loops.csv'
    dtcc_solar.data_io.sunpath_testing(filename2, all_sun_pos, radius)


    plt.show()

