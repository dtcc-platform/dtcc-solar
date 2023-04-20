import os
import pandas as pd
import numpy as np
from pandas import Timestamp
from pprint import pp
from dtcc_solar import data_io
from dtcc_solar import utils
from dtcc_solar.utils import Sun
from typing import List, Dict


# This function reads a *.epw weather file, which contains recorde data for a full year which has been
# compiled by combining data from different months for the years inbetween 2007 and 2021. The time span
# defined in by arguments if then used to obain a sub set of the data for analysis.
def import_weather_data_epw(suns:List[Sun], weather_file:str):

    name_parts = weather_file.split('.')
    if (name_parts[-1] != 'epw'):
        print("The wrong file type was provided. File extension *.epw was expected, but *." + name_parts[-1] + " was given.")
        return None

    # The year is not taken from the *.epw file since it contains a collection of data from 
    # different years typically ranging from 2007 and 2021. Hence the year doesn't really matter.  
    year_str = str(get_year_from_dict_key(suns[0].datetime_str))
    year_dates = []
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
                date_key = year_str + '-' + month + '-' + day + 'T' + hour + ':00:00'
                year_dates.append(date_key)
                direct_normal_radiation.append(float(line_segs[14]))  
                diffuse_horisontal_radiation.append(float(line_segs[15]))
                
            #The first 9 lines contains information of the data structure
            if line[0:12] == 'DATA PERIODS':
                search = True    

    # Restructure the data so that time 24:00:00 is 00:00:00 for the next day.
    year_dates = restructure_time(year_dates)
    year_dates = np.roll(year_dates, 1)
    direct_normal_radiation = np.roll(np.array(direct_normal_radiation), 1)
    diffuse_horisontal_radiation = np.roll(np.array(diffuse_horisontal_radiation), 1)

    sun_index = 0
    epw_index = 0
    for epw_date in year_dates:
        if sun_index < len(suns):
            sun_date = suns[sun_index].datetime_str
            if date_match(epw_date, sun_date):
                suns[sun_index].irradiance_dn = direct_normal_radiation[epw_index]
                suns[sun_index].irradiance_dh = diffuse_horisontal_radiation[epw_index]
                sun_index += 1    
        epw_index += 1

    return suns


# Compare the date stamp from the api data with the date stamp for the generated sun and sky data.
def date_match(api_date, sun_date):
    api_day = api_date[0:10]
    sun_day = sun_date[0:10]
    api_time = api_date[11:19]
    sun_time = api_date[11:19]
    if api_day == sun_day and api_time == sun_time:
        return True
    return False


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

    time_from_str = "2019-03-01 12:00:00"
    time_to_str = "2019-03-03 14:00:00"
    time_from = pd.to_datetime(time_from_str)
    time_to = pd.to_datetime(time_to_str)

    # Run from the location of the file
    weather_file = "../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
    
    suns = utils.create_suns(time_from_str, time_to_str)
    suns = import_weather_data_epw(suns, weather_file)

    pp(suns)


