from pvlib import solarposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import sys
import pathlib
project_dir = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(project_dir)

from dtcc_solar import utils
from dtcc_solar.utils import Vec3

def InitialisePlot(r):
    plt.rcParams['figure.figsize'] = (10,10)
    ax = plt.axes(projection = '3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    return ax

def PlotDayLoopWithText(x, y, z, h, radius, ax, plot_night):
    day_indices = np.where(z > 0)
    night_indices = np.where(z <= 0)
    ax.scatter3D(x[day_indices],y[day_indices],z[day_indices], c=z[day_indices] , cmap = 'autumn_r', vmin = 0, vmax = radius)
    if plot_night:
        ax.scatter3D(x[night_indices],y[night_indices],z[night_indices], color = 'w')
    max_z_indices = np.where(z == np.max(z))
    max_z_index = max_z_indices[0][0]
    text_pos = np.array([x[max_z_index], y[max_z_index], z[max_z_index]])
    ax.text(text_pos[0], text_pos[1], text_pos[2], str(h), fontsize=12)

def PlotDayPath(x, y, z, radius, ax, plot_night):
    day_indices = np.where(z > 0)
    night_indices = np.where(z <= 0)
    z_color = z[day_indices]
    ax.scatter3D(x[day_indices],y[day_indices],z[day_indices], c=z_color , cmap = 'autumn_r', vmin = 0, vmax = radius)

    if plot_night:    
        ax.scatter3D(x[night_indices],y[night_indices],z[night_indices], color = 'w')

def PlotSingleSun(x, y, z, radius, ax):
    z_color = z
    ax.scatter3D(x,y,z, c=z_color , cmap = 'autumn_r', vmin = 0, vmax = radius)        

def PlotDayPath2(sun_pos, radius, ax, plot_night):
    z = sun_pos[:,2]
    day_indices = np.where(z > 0)
    night_indices = np.where(z <= 0)
    zColor = z[day_indices]
    ax.scatter3D(sun_pos[day_indices,0],sun_pos[day_indices,1],sun_pos[day_indices,2], c=zColor , cmap = 'autumn_r', vmin = 0, vmax = radius)
    if plot_night:    
        ax.scatter3D(sun_pos[night_indices,0],sun_pos[night_indices,1],sun_pos[night_indices,2], color = 'w')        
      
class Sunpath():

    def __init__(self, lat, lon, radius, origin):
        self.lat = lat
        self.lon = lon
        self.origin = origin
        self.radius = radius
        self.ax = InitialisePlot(self.radius)
        
    def get_sunpath_hour_loops(self, year, sample_rate , plot_results, plot_night):
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

        xyz = dict.fromkeys([h for h in loop_hours])

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
            xyz[h] = utils.create_list_of_vectors(x[h], y[h], z[h])
            if plot_results:
                PlotDayLoopWithText(x[h], y[h], z[h], h, self.radius, self.ax, plot_night)

        return x,y,z

    def get_sunpath_day_loops(self, dates, step_min, plot_results, plot_night):              

        n = len(dates.values)
        n_evaluation = int(math.ceil(24 * 60 / step_min))
        mat_elev_day = np.zeros((n, n_evaluation+1))
        mat_azim_day = np.zeros((n, n_evaluation+1))
        date_counter = 0
        
        x = dict.fromkeys([d for d in range(0,n)])
        y = dict.fromkeys([d for d in range(0,n)])
        z = dict.fromkeys([d for d in range(0,n)])

        for date in dates.values:
            times = pd.date_range(date, date+pd.Timedelta(str(24) + 'h'), freq= str(step_min) + 'min')
            sol_pos_day = solarposition.get_solarposition(times, self.lat, self.lon)
            rad_elev_day = np.radians(sol_pos_day.apparent_elevation)
            rad_azim_day = np.radians(sol_pos_day.azimuth)
            mat_elev_day[date_counter, :] = rad_elev_day.values
            mat_azim_day[date_counter, :] = rad_azim_day.values
            date_counter = date_counter + 1

        #Plot hourly sun path loops
        for d in range(0,date_counter):
            x[d] = self.radius * np.cos(mat_elev_day[d, :]) * np.cos(-mat_azim_day[d, :]) + self.origin[0]
            y[d] = self.radius * np.cos(mat_elev_day[d, :]) * np.sin(-mat_azim_day[d, :]) + self.origin[1]
            z[d] = self.radius * np.sin(mat_elev_day[d, :]) + self.origin[2]
            if plot_results:
                PlotDayPath(x[d], y[d], z[d], self.radius, self.ax, plot_night)
        
        return x, y, z         

    def get_sun_position_for_a_date(self, date, plot_results, plot_night):

        sun_pos = np.zeros(3)
        sun_pos_day = solarposition.get_solarposition(date, self.lat, self.lon)
        rad_elev_day = np.radians(sun_pos_day.apparent_elevation)
        rad_azim_day = np.radians(sun_pos_day.azimuth)
        elev = rad_elev_day.values
        azim = rad_azim_day.values
        sun_pos[0] = self.radius * np.cos(elev) * np.cos(-azim) + self.origin[0]
        sun_pos[1] = self.radius * np.cos(elev) * np.sin(-azim) + self.origin[1]
        sun_pos[2] = self.radius * np.sin(elev) + self.origin[2]
        if plot_results:  
            PlotSingleSun(sun_pos[0], sun_pos[1], sun_pos[2], self.radius, self.ax)

        sun_positions = np.c_[sun_pos[0], sun_pos[1], sun_pos[2]]

        return sun_positions

    def get_sun_position_for_dates(self, dates, plot_results, plot_night):
        solpos = solarposition.get_solarposition(dates, self.lat, self.lon)
        elev = np.radians(solpos.apparent_elevation.to_list())
        azim = np.radians(solpos.azimuth.to_list())
        x = self.radius * np.cos(elev) * np.cos(-azim) + self.origin[0]
        y = self.radius * np.cos(elev) * np.sin(-azim) + self.origin[1]
        z = self.radius * np.sin(elev) + self.origin[2]
        sun_positions = np.c_[x, y, z]
                
        if plot_results:
            PlotDayPath2(sun_positions, self.radius, self.ax, plot_night)    

        return sun_positions    

    def remove_position_under_horizon(self, horizon_z, sun_positions, dates, dict_keys):
        z = sun_positions[:,2]
        z_mask = (z >= horizon_z)
        sun_positions = sun_positions[z_mask]
        dict_keys = dict_keys[z_mask]   
        dates = dates[z_mask]
        return sun_positions, dates, dict_keys


def run_example():

    os.system('clear')
    print("-------- My Solar Example Started -------")
    origin = np.array([0, 0, 0])
    lat, lon = 51.5, -0.12 #57.71, 11.97 #63.82, 20.26 
    tz = 'Europe/Paris'
    radius = 5 
    analysis = utils.Analyse.Year
    sunpath = Sunpath(lat, lon, radius, origin)
    horizon_z = 0.0

    if(analysis == utils.Analyse.Year):
        year = 2018
        sunpath.get_sunpath_hour_loops(year, 5, True, True)
    elif(analysis == utils.Analyse.Day):
        #dates = pd.to_datetime(['2019-02-21', '2019-06-21', '2019-12-21'])
        dates = pd.date_range(start = '2019-01-01', end = '2019-09-30', freq = '10D')
        sunpath.get_sunpath_day_loops(dates, 10, True, True)
    elif(analysis == utils.Analyse.Time):
        time = pd.to_datetime(['2019-05-30 12:20:00'])
        sunpath.get_sun_position_for_a_date(time, True, True)
    elif(analysis == utils.Analyse.Times):
        dates = pd.to_datetime(['2019-02-21 12:20:00', '2019-06-21 12:20:00', '2019-12-21 12:20:00'])
        sunpath.get_sun_position_for_dates(dates, horizon_z, True, True)    

    plt.show()


if __name__ == "__main__":

    run_example()





