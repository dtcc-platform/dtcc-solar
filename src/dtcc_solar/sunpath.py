import os
import math
import trimesh
import pytz
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar import utils
from dtcc_solar.utils import Vec3, Sun
from dtcc_solar.viewer import Colors
from dtcc_solar.utils import Parameters
from dtcc_solar import weather_data

from pvlib import solarposition
from typing import List, Dict, Any
#from timezonefinder import TimezoneFinder


from pprint import pp

class Sunpath():

    lat: float
    lon: float
    origin: np.ndarray
    radius: float

    def __init__(self, lat:float, lon:float, radius:float):
        self.lat = lat
        self.lon = lon
        self.radius = radius
        self.origin = np.array([0,0,0])
          
    def get_analemmas(self, year:int, sample_rate:int):
        start_date = str(year) + '-01-01 12:00:00'
        end_date = str(year+1) + '-01-01 11:00:00'    
        times = pd.date_range(start = start_date, end = end_date, freq = 'H')

        #Times to evaluate will be 8760 for a normal year and 8764 for leep year  
        times_to_evaluate = len(times.values)
        
        #Number of days will be 365 or 366 at leep year
        num_of_days = int(np.ceil(times_to_evaluate / 24))

        #Get solar position from the pvLib libra
        sol_pos_hour = solarposition.get_solarposition(times, self.lat, self.lon)

        #Reduce the number of days to reduce the density of the sunpath diagram
        days = np.zeros(num_of_days)
        num_evaluated_days = len(days[0:len(days):sample_rate])  

        mat_elev_hour = np.zeros((24, num_evaluated_days))
        mat_azim_hour = np.zeros((24, num_evaluated_days))
        mat_zeni_hour = np.zeros((24, num_evaluated_days))

        loop_hours = np.unique(sol_pos_hour.index.hour)
        x = dict.fromkeys([h for h in loop_hours])
        y = dict.fromkeys([h for h in loop_hours])
        z = dict.fromkeys([h for h in loop_hours])
        analemmas_dict = dict.fromkeys([h for h in loop_hours])

        #Get hourly sun path loops in matrix form and elevaion, azimuth and zenith coordinates
        for hour in loop_hours:
            subset = sol_pos_hour.loc[sol_pos_hour.index.hour == hour, :]
            rad_elev = np.radians(subset.apparent_elevation)
            rad_azim = np.radians(subset.azimuth)
            rad_zeni = np.radians(subset.zenith)
            mat_elev_hour[hour, :] = rad_elev.values[0:len(rad_elev.values):sample_rate]
            mat_azim_hour[hour, :] = rad_azim.values[0:len(rad_azim.values):sample_rate]
            mat_zeni_hour[hour, :] = rad_zeni.values[0:len(rad_zeni.values):sample_rate]

        #Convert hourly sun path loops from spherical to cartesian coordiantes
        for h in range(0,24):
            x[h] = self.radius * np.cos(mat_elev_hour[h, :]) * np.cos(-mat_azim_hour[h, :]) + self.origin[0]
            y[h] = self.radius * np.cos(mat_elev_hour[h, :]) * np.sin(-mat_azim_hour[h, :]) + self.origin[1]
            z[h] = self.radius * np.sin(mat_elev_hour[h, :]) + self.origin[2]                
            analemmas_dict[h] = utils.create_list_of_vectors(x[h], y[h], z[h])
            
        return x,y,z, analemmas_dict

    def get_daypaths(self, dates: pd.DatetimeIndex, minute_step:float):              

        n = len(dates.values)
        n_evaluation = int(math.ceil(24 * 60 / minute_step))
        mat_elev_day = np.zeros((n, n_evaluation+1))
        mat_azim_day = np.zeros((n, n_evaluation+1))
        date_counter = 0
        
        x = dict.fromkeys([d for d in range(0,n)])
        y = dict.fromkeys([d for d in range(0,n)])
        z = dict.fromkeys([d for d in range(0,n)])

        for date in dates.values:
            times = pd.date_range(date, date+pd.Timedelta(str(24) + 'h'), freq= str(minute_step) + 'min')
            sol_pos_day = solarposition.get_solarposition(times, self.lat, self.lon)
            rad_elev_day = np.radians(sol_pos_day.apparent_elevation)
            rad_azim_day = np.radians(sol_pos_day.azimuth)
            mat_elev_day[date_counter, :] = rad_elev_day.values
            mat_azim_day[date_counter, :] = rad_azim_day.values
            date_counter = date_counter + 1

        for d in range(0,date_counter):
            x[d] = self.radius * np.cos(mat_elev_day[d, :]) * np.cos(-mat_azim_day[d, :]) + self.origin[0]
            y[d] = self.radius * np.cos(mat_elev_day[d, :]) * np.sin(-mat_azim_day[d, :]) + self.origin[1]
            z[d] = self.radius * np.sin(mat_elev_day[d, :]) + self.origin[2]
            
        return x, y, z         

    def get_single_sun(self, date: pd.DatetimeIndex):
        sun_pos = np.zeros(3)
        sun_pos_day = solarposition.get_solarposition(date, self.lat, self.lon)
        rad_elev_day = np.radians(sun_pos_day.apparent_elevation)
        rad_azim_day = np.radians(sun_pos_day.azimuth)
        elev = rad_elev_day.values
        azim = rad_azim_day.values
        sun_pos[0] = self.radius * np.cos(elev) * np.cos(-azim) + self.origin[0]
        sun_pos[1] = self.radius * np.cos(elev) * np.sin(-azim) + self.origin[1]
        sun_pos[2] = self.radius * np.sin(elev) + self.origin[2]
        sun_positions = np.c_[sun_pos[0], sun_pos[1], sun_pos[2]]
        return sun_positions

    def get_multiple_suns(self, dict_keys: List[str]):
        dates = pd.to_datetime(dict_keys)
        solpos = solarposition.get_solarposition(dates, self.lat, self.lon)
        elev = np.radians(solpos.apparent_elevation.to_list())
        azim = np.radians(solpos.azimuth.to_list())
        x = self.radius * np.cos(elev) * np.cos(-azim) + self.origin[0]
        y = self.radius * np.cos(elev) * np.sin(-azim) + self.origin[1]
        z = self.radius * np.sin(elev) + self.origin[2]
        sun_positions = np.c_[x, y, z]
                
        return sun_positions    

    def get_suns_positions(self, suns:List[Sun]):

        date_from_str = suns[0].datetime_str
        date_to_str = suns[-1].datetime_str
        dates = pd.date_range(start = date_from_str, end = date_to_str, freq = '1H')

        solpos = solarposition.get_solarposition(dates, self.lat, self.lon)
        elev = np.radians(solpos.apparent_elevation.to_list())
        azim = np.radians(solpos.azimuth.to_list())
        zeni = np.radians(solpos.zenith.to_list())
        x_sun = self.radius * np.cos(elev) * np.cos(-azim) + self.origin[0]
        y_sun = self.radius * np.cos(elev) * np.sin(-azim) + self.origin[1]
        z_sun = self.radius * np.sin(elev) + self.origin[2]

        if(len(suns) == len(dates)):
            for i in range(len(suns)):
                suns[i].position = Vec3(x = x_sun[i], y = y_sun[i], z = z_sun[i])
                suns[i].zenith = zeni[i]
                suns[i].over_horizon = (zeni[i] < math.pi/2)
                suns[i].sun_vec = utils.normalise_vector3(Vec3( x = self.origin[0] - x_sun[i], 
                                                                y = self.origin[1] - y_sun[i], 
                                                                z = self.origin[2] - z_sun[i]))
        else:
            print("Something went wrong in when retrieving solar positions!")
                
        return suns  

    def remove_sun_under_horizon(self, horizon_z, sun_positions, dict_keys):
        z = sun_positions[:,2]
        z_mask = (z >= horizon_z)
        sun_positions = sun_positions[z_mask]
        dict_keys = dict_keys[z_mask]   
        return sun_positions, dict_keys

    def create_sun_timestamps(self, start_date: str, end_date: str):
        time_from = pd.to_datetime(start_date)
        time_to = pd.to_datetime(end_date)
        suns = []
        index = 0
        times = pd.date_range(start = time_from, end = time_to, freq = 'H')
        for time in times:
            sun = Sun(str(time), time, index)
            suns.append(sun)
            index += 1
            
        return suns  

    def create_suns(self, p:Parameters):
        suns = self.create_sun_timestamps(p.start_date, p.end_date)    
        suns = weather_data.get_data(p, suns)                               
        suns = self.get_suns_positions(suns)
        return suns


class SunpathMesh():

    radius:float
    origin: np.ndarray
    sun_meshes: List[Any]
    analemmas_meshes: List[Any]
    daypath_meshes: List[Any]

    def __init__(self, radius:float):
        self.radius = radius
        self.origin = np.array([0,0,0])    

    def get_analemmas_meshes(self):
        return self.analemmas_meshes
    
    def get_daypath_meshes(self):
        return self.daypath_meshes

    def get_sun_meshes(self):
        return self.sun_meshes

    def create_sunpath_diagram(self, suns:List[Sun], sunpath:Sunpath, city_model: SolarEngine, colors:Colors):
        # Create analemmas mesh
        [sunX, sunY, sunZ, analemmas_dict] = sunpath.get_analemmas(2019, 5)
        self.analemmas_meshes = self.create_sunpath_loops(sunX, sunY, sunZ, city_model.sunpath_radius, colors)
        
        # Create mesh for day path
        [sunX, sunY, sunZ] = sunpath.get_daypaths(pd.to_datetime(['2019-06-21', '2019-03-21', '2019-12-21']), 10)
        self.daypath_meshes = self.create_sunpath_loops(sunX, sunY, sunZ, city_model.sunpath_radius, colors)
        
        self.sun_meshes = self.create_solar_spheres(suns, city_model.sun_size)
        
    def create_solar_sphere(self, sunPos, sunSize):
        sunMesh = trimesh.primitives.Sphere(radius = sunSize,  center = sunPos, subdivisions = 4)
        sunMesh.visual.face_colors = [1.0, 0.5, 0, 1.0]
        return sunMesh            

    def create_solar_spheres(self, suns:List[Sun], sunSize):
        sunMeshes = []
        for i in range(0,len(suns)):
            if(suns[i].over_horizon):
                sun_pos = utils.convert_vec3_to_ndarray(suns[i].position)
                sunMesh = trimesh.primitives.Sphere(radius = sunSize, center = sun_pos, subdivisions = 1)
                sunMesh.visual.face_colors = [1.0, 0.5, 0, 1.0]
                sunMeshes.append(sunMesh)
        return sunMeshes

    def create_sunpath_loops(self, x, y, z, radius, colors:Colors):
        path_meshes = []
        for h in x:
            vs = np.zeros((len(x[h])+1, 3))
            vi = np.zeros((len(x[h])),dtype=int)
            lines = []
            path_colors = []
            
            for i in range(0, len(x[h])):
                sunPos = [x[h][i], y[h][i], z[h][i]]
                vs[i,:] = sunPos
                vi[i] = i
                index2 = i + 1
                color = colors.get_blended_color_yellow_red(radius, z[h][i])
                path_colors.append(color)
                line = trimesh.path.entities.Line([i, index2])
                lines.append(line)

            vs[len(x[h]),:] = vs[0,:]     

            path = trimesh.path.Path3D(entities=lines, vertices=vs, colors= path_colors)
            path_meshes.append(path)

        return path_meshes        


class SunpathVis():

    def __init__(self):
        pass   

    def initialise_plot(self, r, title):
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

    def plot_imported_sunpath_diagarm(self, pts, radius, ax, cmap):
        
        n = int(0.5 * len(pts[0]))
        x,y,z = [],[],[]
        for hour in pts:
            for i in range(n):
                pt = pts[hour][i]
                x.append(pt.x)
                y.append(pt.y)
                z.append(pt.z)    
        
        ax.scatter3D(x, y, z, c=z, cmap = cmap, vmin = 0, vmax = radius)

    def plot_analemmas(self, all_sun_pos: Dict[int, List[Vec3]],radius, ax, plot_night, cmap, gmt_diff):
        
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
            local_hour = SunpathUtils.convert_utc_to_local_time(utc_hour, gmt_diff)
            text_pos_max = np.array([x[z_max_indices[i]], y[z_max_indices[i]], z[z_max_indices[i]]])
            ax.text(text_pos_max[0], text_pos_max[1], text_pos_max[2], str(local_hour), fontsize=12)

    def plot_daypath(self, x_dict, y_dict, z_dict, radius, ax, plot_night):

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

    def plot_single_sun(self, x, y, z, radius, ax):
        z_color = z
        ax.scatter3D(x,y,z, c=z_color , cmap = 'autumn_r', vmin = 0, vmax = radius)        

    def plot_multiple_suns(self, sun_pos, radius, ax, plot_night):
        z = sun_pos[:,2]
        day_indices = np.where(z > 0)
        night_indices = np.where(z <= 0)
        zColor = z[day_indices]
        ax.scatter3D(sun_pos[day_indices,0],sun_pos[day_indices,1],sun_pos[day_indices,2], c=zColor , cmap = 'autumn_r', vmin = 0, vmax = radius)
        if plot_night:    
            ax.scatter3D(sun_pos[night_indices,0],sun_pos[night_indices,1],sun_pos[night_indices,2], color = 'w')        


class SunpathUtils():

    def __init__(self):  
        pass  

    @staticmethod    
    def convert_utc_to_local_time(utc_h, gmt_diff):

        local_h = utc_h + gmt_diff
        if(local_h < 0):
            local_h = 24 + local_h
        elif (local_h > 23):    
            local_h = local_h - 24

        return local_h
    
    @staticmethod
    def convert_local_time_to_utc(local_h, gmt_diff):

        utc_h = local_h - gmt_diff
        if(utc_h < 0):
            utc_h = 24 + utc_h
        elif (utc_h > 23):    
            utc_h = utc_h - 24

        return utc_h

    @staticmethod
    def get_timezone_from_long_lat(lat, long):
        tf = 0 #= TimezoneFinder() 
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

    pass





