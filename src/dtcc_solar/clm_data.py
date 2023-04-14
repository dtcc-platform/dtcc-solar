import os
import pandas as pd
import numpy as np
from pprint import pp
from dtcc_solar import utils


# This function reads a *.clm weather file, which contains recorde data for a full year which has been
# compiled by combining data from different months for the years inbetween 2007 and 2021. The time span
# defined in by arguments if then used to obain a sub set of the data for analysis. 
def import_weather_data_clm(start_date:pd.Timestamp, end_date:pd.Timestamp, weather_file:str):
    
    name_parts = weather_file.split('.')
    if (name_parts[-1] != 'clm'):
        print("The wrong file type was provided. File extension *.clm was expected, but *." + name_parts[-1] + " was given.")
        return None

    line_index = 0
    data_counter = 0
    #The data file contains weather data for an entire year
    year_dates = pd.date_range(start = '2019-01-01 00:00:00', end = '2019-12-31 23:59:00', freq = '1H')
    year_dict_keys = np.array([str(d) for d in year_dates])
    year_dict_keys = utils.format_dict_keys(year_dict_keys)
    year_weather_data = dict.fromkeys(year_dict_keys)

    with open(weather_file, 'r') as f:
        for line in f:
            #The first 12 lines contains information of the data structure
            if line_index > 12 and line[0] != '*':
               year_weather_data[year_dict_keys[data_counter]] = line2numbers(line)     
               data_counter += 1     
            
            line_index += 1
    
    sub_dates = pd.date_range(start = start_date, end = end_date, freq = '1H')
    sub_dict_keys = np.array([str(d) for d in sub_dates])
    sub_dict_keys = utils.format_dict_keys(sub_dict_keys)        
    sub_weather_data = utils.get_weather_data_subset(year_weather_data, sub_dict_keys)    
    return sub_weather_data, sub_dict_keys


def line2numbers(line):
    keys = ['normal_irradiance', 'horizontal_irradiance']
    adict = dict.fromkeys([k for k in keys])
    line = line.replace('\n', '')
    linesegs = line.split(',')
    if(len(linesegs) == 6):
        adict['normal_irradiance'] = float(linesegs[2])
        adict['horizontal_irradiance'] = float(linesegs[0])
        
    return adict


if __name__ == "__main__":

    os.system('clear')    
    print("--------------------- Running main function for CLM data import -----------------------")

    # Run from the location of the file
    weather_file = "../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
    time_from = pd.to_datetime("2019-03-01 00:00:00")
    time_to = pd.to_datetime("2019-03-25 23:00:00")

    [sub_weather_data, sub_dict_keys] = import_weather_data_clm(time_from, time_to, weather_file)

    pp(sub_weather_data, width = 100)

    pp(sub_dict_keys, width = 100)

