import os
import pandas as pd
import numpy as np
from pandas import Timestamp
from pprint import pp
from dtcc_solar import data_io
from dtcc_solar import utils


# This function reads a *.epw weather file, which contains recorde data for a full year which has been
# compiled by combining data from different months for the years inbetween 2007 and 2021. The time span
# defined in by arguments if then used to obain a sub set of the data for analysis.
def import_weather_data_epw(start_date:pd.Timestamp, end_date:pd.Timestamp, weather_file:str):

    name_parts = weather_file.split('.')
    if (name_parts[-1] != 'epw'):
        print("The wrong file type was provided. File extension *.epw was expected, but *." + name_parts[-1] + " was given.")
        return None

    # The year is not taken from the *.epw file since it contains a collection of data from 
    # different years typically ranging from 2007 and 2021. Hence the year doesn't really matter.  
    year = str(get_year_from_dict_key(str(start_date)))
    year_dict_keys = []
    direct_normal_radiation = []
    diffuse_horisontal_radiation = []
    search = False

    with open(weather_file, 'r') as f:
        for line in f:
            if search:
                line_segs = line.split(',')
                month = make_double_digit_str(line_segs[1])
                day = make_double_digit_str(line_segs[2])
                hour = make_double_digit_str(line_segs[3])
                date_key = year + '-' + month + '-' + day + 'T' + hour + ':00:00'
                year_dict_keys.append(date_key)
                direct_normal_radiation.append(float(line_segs[14]))  
                diffuse_horisontal_radiation.append(float(line_segs[15]))
                
            #The first 9 lines contains information of the data structure
            if line[0:12] == 'DATA PERIODS':
                search = True    

    # Restructure the data so that time 24:00:00 is 00:00:00 for the next day.
    year_dict_keys = restructure_time(year_dict_keys)
    year_dict_keys = np.roll(year_dict_keys, 1)
    direct_normal_radiation = np.roll(np.array(direct_normal_radiation), 1)
    diffuse_horisontal_radiation = np.roll(np.array(diffuse_horisontal_radiation), 1)
    
    year_weather_data = dict.fromkeys(year_dict_keys)
    for i, key in enumerate(year_dict_keys):
        d_type = ['normal_irradiance', 'horizontal_irradiance']
        a_dict = dict.fromkeys([d for d in d_type])
        a_dict['normal_irradiance'] = direct_normal_radiation[i] 
        a_dict['horizontal_irradiance'] = diffuse_horisontal_radiation[i]
        year_weather_data[key] = a_dict

    sub_dates = pd.date_range(start = start_date, end = end_date, freq = '1H')
    sub_dict_keys = np.array([str(d) for d in sub_dates])
    sub_dict_keys = utils.format_dict_keys(sub_dict_keys)    
    
    sub_weather_data = utils.get_weather_data_subset(year_weather_data, sub_dict_keys)    
    return sub_weather_data, sub_dict_keys

# The data in epw files are structured such that time 00:00:00 for a tuesday is called 24:00:00 on the monday instead.
# This function reorganises the data to follow the 00:00:00 standard instead.  
def restructure_time(year_dict_keys):
    new_years_dict_key = np.repeat("0000-00-00T00:00:00", len(year_dict_keys))
    counter = 0
    for key in year_dict_keys:
        hour = get_hour_from_dict_key(key)
        if(hour == 24):
            year = get_year_from_dict_key(key)
            month = get_month_from_dict_key(key)
            day = get_day_from_dict_key(key)
            new_date = increment_date(year, month, day+1)
            new_time = "00:00:00"
            new_key = new_date + 'T' + new_time
            new_years_dict_key[counter] = new_key
        else:
            new_years_dict_key[counter] = key
        counter += 1     

    return new_years_dict_key

def increment_date(year, month, day):
    #Last day of the year.
    if(month == 12 and day == 31):
        t = pd.to_datetime(str(year) + '-' + '01' + '-' + '01')
        parts = str(t).split(' ')
        return parts[0]
    else:
        try:
            t = pd.to_datetime(str(year) + '-' + str(month) + '-' + str(day))
            parts = str(t).split(' ')
            return parts[0]
        except:
            day = 1
            month += 1
            if(month > 12):
                month = 1

            t = pd.to_datetime(str(year) + '-' + str(month) + '-' + str(day))
            parts = str(t).split(' ')
            return parts[0]
        
def get_hour_from_dict_key(dict_key):
    hour = int(dict_key[11:13])
    return hour

def get_day_from_dict_key(dict_key):
    day = int(dict_key[8:10])
    return day

def get_year_from_dict_key(dict_key):
    year = int(dict_key[0:4])
    return year

def get_month_from_dict_key(dict_key):
    month = int(dict_key[5:7])
    return month

def make_double_digit_str(s:str):
    if len(s) == 1:
        s = '0' + s
    return s     


if __name__ == "__main__":

    os.system('clear')    
    print("--------------------- Running main function for EPW data import -----------------------")

    # Run from the location of the file
    weather_file = "../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
    time_from = pd.to_datetime("2019-03-01 12:00:00")
    time_to = pd.to_datetime("2019-03-03 14:00:00")

    [sub_weather_data, sub_dict_keys] = import_weather_data_epw(time_from, time_to, weather_file)


