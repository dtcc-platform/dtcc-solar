from pvlib import solarposition
import pandas as pd
import numpy as np
import os
import math
import sys
import pathlib
project_dir = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(project_dir)

from dtcc_solar import utils
from dtcc_solar.utils import Vec3

class Sunpath():

    lat: float
    lon: float
    origin: np.ndarray
    radius: float

    def __init__(self, lat:float, lon:float, radius:float, origin:np.ndarray):
        self.lat = lat
        self.lon = lon
        self.origin = origin
        self.radius = radius
        
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
        pos_dict = dict.fromkeys([h for h in loop_hours])

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
            pos_dict[h] = utils.create_list_of_vectors(x[h], y[h], z[h])
            
        return x,y,z, pos_dict

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

    def get_multiple_suns(self, dates: pd.DatetimeIndex):
        solpos = solarposition.get_solarposition(dates, self.lat, self.lon)
        elev = np.radians(solpos.apparent_elevation.to_list())
        azim = np.radians(solpos.azimuth.to_list())
        x = self.radius * np.cos(elev) * np.cos(-azim) + self.origin[0]
        y = self.radius * np.cos(elev) * np.sin(-azim) + self.origin[1]
        z = self.radius * np.sin(elev) + self.origin[2]
        sun_positions = np.c_[x, y, z]
                
        return sun_positions    

    def remove_sun_under_horizon(self, horizon_z, sun_positions, dates, dict_keys):
        z = sun_positions[:,2]
        z_mask = (z >= horizon_z)
        sun_positions = sun_positions[z_mask]
        dict_keys = dict_keys[z_mask]   
        dates = dates[z_mask]
        return sun_positions, dates, dict_keys



if __name__ == "__main__":

    pass





