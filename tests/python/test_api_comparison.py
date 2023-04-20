import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pp
from dtcc_solar import utils
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import AnalysisType, Parameters, Sun
from typing import List, Dict, Any
from dtcc_solar import weather_data as wd

class TestWeatherDataComparison:

    lat: float
    lon: float

    def setup_method(self):
        self.lon = 16.158
        self.lat = 58.5812
        self.w_file_clm = "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
        self.w_file_epw = "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
        self.file_name = '../data/models/CitySurfaceS.stl'

    def test_compare_dict_keys(self):

        start_date = "2019-01-01 00:00:00"
        end_date = "2019-12-05 00:00:00"
        sunpath = Sunpath(self.lat, self.lon, 1.0)
        a_type = AnalysisType.sun_raycasting
        
        p_smhi = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 1, 1, False, start_date, end_date, self.w_file_clm, 2)
        p_meteo = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 2, 1, False, start_date, end_date, self.w_file_clm, 2)
        p_clm = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 3, 1, False, start_date, end_date, self.w_file_clm, 2)
        p_epw = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 4, 1, False, start_date, end_date, self.w_file_epw, 2)

        suns_date = utils.create_sun_dates(start_date, end_date)    
        
        suns_smhi = wd.get_weather_data(p_smhi, suns_date)                               
        suns_smhi = sunpath.get_suns_positions(suns_smhi)

        suns_meteo = wd.get_weather_data(p_meteo, suns_date)                               
        suns_meteo = sunpath.get_suns_positions(suns_meteo)

        suns_clm = wd.get_weather_data(p_clm, suns_date)                               
        suns_clm = sunpath.get_suns_positions(suns_clm)
        
        suns_epw = wd.get_weather_data(p_epw, suns_date)                               
        suns_epw = sunpath.get_suns_positions(suns_epw)

        
        if (len(suns_smhi) != len(suns_meteo) or  
            len(suns_smhi) != len(suns_clm) or
            len(suns_smhi) != len(suns_epw)):
            print("Keys missmatch")
            return False
        
        for i in range(len(suns_smhi)):
            if (suns_smhi[i].datetime_str != suns_meteo[i].datetime_str or 
                suns_smhi[i].datetime_str != suns_clm[i].datetime_str or
                suns_smhi[i].datetime_str != suns_epw[i].datetime_str ):       
                print("Keys missmatch")
                return False

        print("Keys matching asserted")
        assert True

    def test_calculate_monthly_average_data(self):

        start_date = "2019-01-01 00:00:00"
        end_date = "2019-12-31 00:00:00"
        sunpath = Sunpath(self.lat, self.lon, 1.0)
        a_type = AnalysisType.sun_raycasting
        
        p_smhi = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 1, 1, False, start_date, end_date, self.w_file_clm, 2)
        p_meteo = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 2, 1, False, start_date, end_date, self.w_file_clm, 2)
        p_clm = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 3, 1, False, start_date, end_date, self.w_file_clm, 2)
        p_epw = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 4, 1, False, start_date, end_date, self.w_file_epw, 2)

        suns_date = utils.create_sun_dates(start_date, end_date)    
        
        suns_smhi = wd.get_weather_data(p_smhi, suns_date)                               
        suns_smhi = sunpath.get_suns_positions(suns_smhi)
        [w_data_normal_smhi, w_data_horizontal_smhi] = format_data_per_day_suns(suns_smhi)

        smhi_normal_avrg = get_monthly_average_data(w_data_normal_smhi)
        smhi_horizon_avrg = get_monthly_average_data(w_data_horizontal_smhi)

        suns_meteo = wd.get_weather_data(p_meteo, suns_date)                               
        suns_meteo = sunpath.get_suns_positions(suns_meteo)
        [w_data_normal_meteo, w_data_horizontal_meteo] = format_data_per_day_suns(suns_meteo)

        meteo_normal_avrg = get_monthly_average_data(w_data_normal_meteo)
        meteo_horizon_avrg = get_monthly_average_data(w_data_horizontal_meteo)

        suns_clm = wd.get_weather_data(p_clm, suns_date)                               
        suns_clm = sunpath.get_suns_positions(suns_clm)
        [w_data_normal_clm, w_data_horizontal_clm] = format_data_per_day_suns(suns_clm)

        clm_normal_avrg = get_monthly_average_data(w_data_normal_clm)
        clm_horizon_avrg = get_monthly_average_data(w_data_horizontal_clm)

        suns_epw = wd.get_weather_data(p_epw, suns_date)                               
        suns_epw = sunpath.get_suns_positions(suns_epw)
        [w_data_normal_epw, w_data_horizontal_epw] = format_data_per_day_suns(suns_epw)
        
        epw_normal_avrg = get_monthly_average_data(w_data_normal_epw)
        epw_horizon_avrg = get_monthly_average_data(w_data_horizontal_epw)

        plt.figure(1)
        ax_1 = plt.subplot(2,4,1)
        title = 'SMHI Monthly average normal irradiance'
        plot_montly_average(title, ax_1, smhi_normal_avrg, 'Time of day', 'W/m2', 0, 1000)

        ax_2 = plt.subplot(2,4,5)
        title = 'SMHI Monthly average horisontal irradiance'
        plot_montly_average(title, ax_2, smhi_horizon_avrg, 'Time of day', 'W/m2', 0, 1000)

        ax_3 = plt.subplot(2,4,2)
        title = 'Meteo Monthly average normal irradiance'
        plot_montly_average(title, ax_3, meteo_normal_avrg, 'Time of day', 'W/m2', 0, 1000)

        ax_4 = plt.subplot(2,4,6)
        title = 'Meteo Monthly average horisontal irradiance'
        plot_montly_average(title, ax_4, meteo_horizon_avrg, 'Time of day', 'W/m2', 0, 1000)

        ax_5 = plt.subplot(2,4,3)
        title = 'CLM Monthly average normal irradiance'
        plot_montly_average(title, ax_5, clm_normal_avrg, 'Time of day', 'W/m2', 0, 1000)

        ax_6 = plt.subplot(2,4,7)
        title = 'CLM Monthly average horisontal irradiance'
        plot_montly_average(title, ax_6, clm_horizon_avrg, 'Time of day', 'W/m2', 0, 1000)

        ax_7 = plt.subplot(2,4,4)
        title = 'EPW Monthly average normal irradiance'
        plot_montly_average(title, ax_7, epw_normal_avrg, 'Time of day', 'W/m2', 0, 1000)

        ax_8 = plt.subplot(2,4,8)
        title = 'EPW Monthly average horisontal irradiance'
        plot_montly_average(title, ax_8, epw_horizon_avrg, 'Time of day', 'W/m2', 0, 1000)


        plt.show()


    def test_compare_weather_data(self):
        
        start_date = "2019-01-01 00:00:00"
        end_date = "2019-12-31 00:00:00"
        sunpath = Sunpath(self.lat, self.lon, 500.0)
        a_type = AnalysisType.sun_raycasting
        
        p_smhi = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 1, 1, False, start_date, end_date, self.w_file_clm, 2)
        p_meteo = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 2, 1, False, start_date, end_date, self.w_file_clm, 2)
        p_clm = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 3, 1, False, start_date, end_date, self.w_file_clm, 2)
        p_epw = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 4, 1, False, start_date, end_date, self.w_file_epw, 2)

        suns_date = utils.create_sun_dates(start_date, end_date)    
        
        suns_smhi = wd.get_weather_data(p_smhi, suns_date)                               
        suns_smhi = sunpath.get_suns_positions(suns_smhi)
        [w_data_normal_smhi, w_data_horizontal_smhi] = format_data_per_day_suns(suns_smhi)

        suns_meteo = wd.get_weather_data(p_meteo, suns_date)                               
        suns_meteo = sunpath.get_suns_positions(suns_meteo)
        [w_data_normal_meteo, w_data_horizontal_meteo] = format_data_per_day_suns(suns_meteo)

        suns_clm = wd.get_weather_data(p_clm, suns_date)                               
        suns_clm = sunpath.get_suns_positions(suns_clm)
        [w_data_normal_clm, w_data_horizontal_clm] = format_data_per_day_suns(suns_clm)

        suns_epw = wd.get_weather_data(p_epw, suns_date)                               
        suns_epw = sunpath.get_suns_positions(suns_epw)
        [w_data_normal_epw, w_data_horizontal_epw] = format_data_per_day_suns(suns_epw)

        if (len(suns_smhi) != len(suns_meteo)):
            print("SMHI and Open Meteo data format missmatch")
            return None

        log_normal = w_data_normal_smhi.copy()
        log_horizontal = w_data_normal_smhi.copy()

        np.seterr(divide = 'ignore') 
        
        for key in w_data_normal_smhi:
            a = np.array(w_data_normal_smhi[key])
            b = np.array(w_data_normal_meteo[key])    
            log_n = np.divide(a, b, out = np.zeros_like(a), where = b != 0)
            log_n = np.log(log_n)
            log_normal[key] = log_n

            a = np.array(w_data_horizontal_smhi[key])
            b = np.array(w_data_horizontal_meteo[key])    
            log_h = np.divide(a, b, out = np.zeros_like(a), where = b != 0)
            log_h = np.log(log_h)    
            log_horizontal[key] = log_h

        np.seterr(divide = 'warn')

        x_data = np.array(range(0,24))

        print("Weather data computation asserted")

        plt.figure(1)
        ax_1 = plt.subplot(2,4,1)
        plt.title("SMHI Normal irradiance")
        subplot_setup(ax_1, 'Time of day', 'W/m2', 0, 23, 0, 1000)
        for key in w_data_horizontal_smhi:
            y_data_normal = w_data_normal_smhi[key]
            plt.plot(x_data,y_data_normal)
        
        ax_2= plt.subplot(2,4,5)
        plt.title("SMHI Horizontal irradiance")
        subplot_setup(ax_2, 'Time of day', 'W/m2', 0, 23, 0, 1000)
        for key in w_data_horizontal_smhi:
            y_data_horizontal = w_data_horizontal_smhi[key]
            plt.plot(x_data,y_data_horizontal)
                    
        ax_3 = plt.subplot(2,4,2)
        plt.title("Open Meteo Normal irradiance")
        subplot_setup(ax_3, 'Time of day', 'W/m2', 0, 23, 0, 1000) 
        for key in w_data_horizontal_meteo:
            y_data_normal = w_data_normal_meteo[key]
            plt.plot(x_data,y_data_normal)
        
        ax_4= plt.subplot(2,4,6)
        plt.title("Open Meteo Horizontal irradiance")
        subplot_setup(ax_4, 'Time of day', 'W/m2', 0, 23, 0, 1000)
        for key in w_data_horizontal_meteo:
            y_data_horizontal = w_data_horizontal_meteo[key]
            plt.plot(x_data,y_data_horizontal)

        ax_5 = plt.subplot(2,4,3)
        plt.title("CLM-file Normal irradiance")
        subplot_setup(ax_5, 'Time of day', 'W/m2', 0, 23, 0, 1000) 
        for key in w_data_horizontal_meteo:
            y_data_normal = w_data_normal_clm[key]
            plt.plot(x_data,y_data_normal)
        
        ax_6= plt.subplot(2,4,7)
        plt.title("CLM-file Horizontal irradiance")
        subplot_setup(ax_6, 'Time of day', 'W/m2', 0, 23, 0, 1000)
        for key in w_data_horizontal_meteo:
            y_data_horizontal = w_data_horizontal_clm[key]
            plt.plot(x_data,y_data_horizontal)

        ax_7 = plt.subplot(2,4,4)
        plt.title("EPW-file Normal irradiance")
        subplot_setup(ax_7, 'Time of day', 'W/m2', 0, 23, 0, 1000) 
        for key in w_data_horizontal_meteo:
            y_data_normal = w_data_normal_epw[key]
            plt.plot(x_data,y_data_normal)
        
        ax_8= plt.subplot(2,4,8)
        plt.title("EPW-file Horizontal irradiance")
        subplot_setup(ax_8, 'Time of day', 'W/m2', 0, 23, 0, 1000)
        for key in w_data_horizontal_meteo:
            y_data_horizontal = w_data_horizontal_epw[key]
            plt.plot(x_data,y_data_horizontal)

        plt.figure(2)    
        ax_10 = plt.subplot(1,2,1)
        plt.title("Normal irradinance - Log(SMHI/Meteo)")
        ax_10.set_xlabel('Time of day')
        ax_10.set_xticks(range(0,23)) 
        for key in w_data_horizontal_meteo:
            y_data_normal = log_normal[key]
            plt.plot(x_data,y_data_normal)
        
        ax_11= plt.subplot(1,2,2)
        plt.title("Horizontal irradiance - Log(SMHI/Meteo)")
        ax_11.set_xlabel('Time of day')
        ax_11.set_xticks(range(0,23))
        for key in w_data_horizontal_meteo:
            y_data_horizontal = log_horizontal[key]
            plt.plot(x_data,y_data_horizontal)

        plt.show()
        pass


def get_monthly_average_data(data_dict):
    
    avrg_data = np.zeros((12,24))
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    key_list = list(data_dict.keys())

    for i,v in enumerate(key_list):
        month = get_month_from_dict_key(key_list[i])
        for j in range(0,23):
            avrg_data[month-1, j] += data_dict[key_list[i]][j] / days_per_month[month-1]    
    
    return avrg_data


def get_month_from_dict_key(dict_key):
    month = int(dict_key[5:7])
    return month

def plot_montly_average(title, ax, data, x_label, y_label, y_min, y_max):
    x_data = range(0,24)
    plt.title(title)
    subplot_setup(ax, x_label, y_label, 0, 23, y_min, y_max)
    for i in range(0,12):
        y_data_normal = data[i,:]
        plt.plot(x_data,y_data_normal)

    pass


def subplot_setup(ax, x_label, y_label, x_min, x_max, y_min, y_max):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(range(0,23)) 


def format_data_per_day_suns(suns:List[Sun]):

    date_keys = []
    for sun in suns:
        key = sun.datetime_str
        date = key[0:10]
        date_keys.append(date)

    normal_irr = dict.fromkeys(date_keys)
    horizon_irr = dict.fromkeys(date_keys)

    data_ni = []
    data_hi = []
    for i in range(len(suns)):
        key = suns[i].datetime_str
        date_key = key[0:10]
        hour = suns[i].datetime_ts.hour
        data_ni.append(suns[i].irradiance_dn)
        data_hi.append(suns[i].irradiance_di)

        if(hour == 23):
            normal_irr[date_key] = data_ni
            horizon_irr[date_key] = data_hi
            data_ni = []
            data_hi = []

    empty_keys = [k for k, v in normal_irr.items() if v == None]
    if(len(empty_keys)):
        normal_irr.pop(empty_keys[0])
        
    empty_keys = [k for k, v in horizon_irr.items() if v == None]
    if(len(empty_keys)):
        horizon_irr.pop(empty_keys[0])

    return normal_irr, horizon_irr

# To avoid error message "RuntimeWarning: divide by zero encountered in np.log(log_h)"
def safe_log10(x, eps=1e-10):     
    result = np.where(x > eps, x, -10)     
    np.log10(result, out=result, where=result > 0)     
    return result


if __name__ == "__main__":

    os.system('clear')
    
    print("--------------------- API Comparision test started -----------------------")

    test = TestWeatherDataComparison()
    test.setup_method()
    #test.test_compare_dict_keys()
    #test.test_calculate_monthly_average_data()
    test.test_compare_weather_data()
    
    pass






