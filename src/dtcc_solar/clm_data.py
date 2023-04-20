import os
import pandas as pd
import numpy as np
from pprint import pp
from typing import List, Dict
from dtcc_solar import utils
from dtcc_solar.utils import Sun, Sky

# This function reads a *.clm weather file, which contains recorde data for a full year which has been
# compiled by combining data from different months for the years inbetween 2007 and 2021. The time span
# defined in by arguments if then used to obain a sub set of the data for analysis. 
def import_weather_data_clm(suns:List[Sun], skys:List[Sky], weather_file:str):

    name_parts = weather_file.split('.')
    if (name_parts[-1] != 'clm'):
        print("The wrong file type was provided. File extension *.clm was expected, but *." + name_parts[-1] + " was given.")
        return None

    line_index = 0
    data_counter = 0
    year = suns[0].datetime_ts.year
    full_year_start_date = str(year) + "-01-01 00:00:00"
    full_year_end_date = str(year) + "-12-31 23:59:00"

    #The data file contains weather data for an entire year
    year_dates = pd.date_range(start = full_year_start_date, end = full_year_end_date, freq = '1H')
    year_dict_keys = np.array([str(d) for d in year_dates])
    year_dict_keys = utils.format_dict_keys(year_dict_keys)
    year_normal_irradiance = dict.fromkeys(year_dict_keys)
    year_diffuse_irradiance = dict.fromkeys(year_dict_keys)

    with open(weather_file, 'r') as f:
        for line in f:
            #The first 12 lines contains information of the data structure
            if line_index > 12 and line[0] != '*':
               [normal_irradiance, diffuse_irradiance] = line2numbers(line)
               year_normal_irradiance[year_dict_keys[data_counter]] = normal_irradiance
               year_diffuse_irradiance[year_dict_keys[data_counter]] = diffuse_irradiance
               data_counter += 1         
            line_index += 1

    all_dates = list(year_normal_irradiance.keys())
    sun_index = 0

    for clm_date in all_dates:
        if sun_index < len(suns):
            sun_date = suns[sun_index].datetime_str
            if date_match(clm_date, sun_date):
                suns[sun_index].irradiance_dn = year_normal_irradiance[clm_date]
                skys[sun_index].irradiance_dh = year_diffuse_irradiance[clm_date]
                sun_index += 1         

    return suns, skys

# Compare the date stamp from the api data with the date stamp for the generated sun and sky data.
def date_match(api_date, sun_date):
    api_day = api_date[0:10]
    sun_day = sun_date[0:10]
    api_time = api_date[11:19]
    sun_time = sun_date[11:19]
    if api_day == sun_day and api_time == sun_time:
        return True
    return False


def line2numbers(line):
    line = line.replace('\n', '')
    linesegs = line.split(',')
    if(len(linesegs) == 6):
        normal_irradiance = float(linesegs[2])
        diffuse_horizontal_irradiance = float(linesegs[0])
        
    return normal_irradiance, diffuse_horizontal_irradiance


if __name__ == "__main__":

    os.system('clear')    
    print("--------------------- Running main function for CLM data import -----------------------")

    time_from_str = "2019-03-01 00:00:00"
    time_to_str = "2019-03-25 23:00:00"
    time_from = pd.to_datetime(time_from_str)
    time_to = pd.to_datetime(time_to_str)

    [suns, skys] = utils.create_sun_and_sky(time_from_str, time_to_str)

    # Run from the location of the file
    weather_file = "../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"

    [suns, skys] = import_weather_data_clm(suns, skys, weather_file)

    pp(suns)

    #pp(skys)


