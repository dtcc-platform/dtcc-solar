import requests
import os
import pandas as pd
import numpy as np
from pandas import Timestamp
from pprint import pp
from dtcc_solar import data_io
from dtcc_solar import utils
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from typing import Dict, List

def get_data_from_api_call(lon:float, lat:float, date_from:Timestamp, date_to:Timestamp):

    date_from_str, time_from_str = timestamp_str(date_from)
    date_to_str, time_to_str = timestamp_str(date_to)

    # API call cannot handle time as part of the date variable. So the data is first
    # collected based on the date and then a subset is retriwed which also considers
    # the time in the date variable. 
    url_1 = "https://archive-api.open-meteo.com/v1/archive?"
    url_2 = "latitude=" + str(lat) + "&longitude=" + str(lon)
    url_3 = "&start_date=" + date_from_str + "&end_date=" + date_to_str
    url_4 = "&hourly=direct_normal_irradiance"
    url_5 = "&hourly=direct_radiation" #diffuse_radiation"
    
    # Direct normal irradiance:
    # Average of the preceding hour on the normal plane (perpendicular to the sun)
    # Units W/m2
    url_ni = url_1 + url_2 + url_3 + url_4
    normal_irradiance = requests.get(url_ni)
    status_ni = normal_irradiance.status_code

    # Diffuse solar radiation
    # Diffuse solar radiation as average of the preceding hour
    # Units W/m2
    url_hr = url_1 + url_2 + url_3 + url_5
    horizon_radiation = requests.get(url_hr)
    status_hr = horizon_radiation.status_code
    
    if (status_ni == 200 and status_hr == 200):
        dict_keys = normal_irradiance.json()["hourly"]["time"]
        dict_keys = format_dict_keys(dict_keys)
        
        if(utils.check_dict_keys_format(dict_keys)):
            normal_irradiance_hourly = normal_irradiance.json()["hourly"]["direct_normal_irradiance"]
            horizon_radiation_hourly = horizon_radiation.json()["hourly"]["direct_radiation"]#["diffuse_radiation"]

            w_data_dict = dict.fromkeys(dict_keys)
            for i, key in enumerate(dict_keys):
                ni = normal_irradiance_hourly[i]
                hr = horizon_radiation_hourly[i]
                types = ['normal_irradiance', 'horizontal_irradiance']
                data_dict = dict.fromkeys([t for t in types])
                data_dict['normal_irradiance'] = ni
                data_dict['horizontal_irradiance'] = hr
                w_data_dict[key] = data_dict

            # Get time dependent sub set of data.
            [w_data_dict_sub, dict_keys_sub] = get_data_subset(time_from_str, time_to_str, dict_keys, w_data_dict)

            return w_data_dict_sub, dict_keys_sub
        else:
            print("Problem with the format of the weather data dict keys!")
            return None

    elif (status_ni == 400 or status_hr == 400):
        print("Open Meteo HTTP status code 400:")
        print("The request points to a resource that do not exist." +
        "This might happen if you query for a station that do" +
        "not produce data for a specific parameter or you are" +
        "using a deprecated version of the API that has been" +
        "removed. It might also be the case that the specified" +
        "station do not have any data for the latest hour.")
        return False
        
# The dict key format is (without blank spaces):
# Year - Month - Day T hour : minute : second
# Example 2018-03-23T12:00:00      
def format_dict_keys(dict_keys: List[str]):
    for i in range(len(dict_keys)):
        dict_keys[i] += ":00"
    return dict_keys

def format_date_range(dict_keys_subset: pd.DatetimeIndex):

    dict_keys_list = []
    for key in dict_keys_subset:
        dict_keys_list.append(key)
    pass

# Based on the assumption that time_from is part of the first day
# and time_to is part of the last day this function returns the 
# time dependent subset of the data from the API call
def get_data_subset(time_from:str, time_to:str, dict_keys: List[str], w_data_dict):
    
    hour_from = int(time_from[0:2])
    hour_to = int(time_to[0:2])
    # Remove the first and last  
    dict_keys_subset = dict_keys[hour_from: (len(dict_keys)-(23 - hour_to))]
    w_data_dict_subset = dict.fromkeys(dict_keys_subset)
    for key in dict_keys_subset:
        w_data_dict_subset[key] = w_data_dict[key]

    dict_keys_subset = np.array(dict_keys_subset)

    return w_data_dict_subset, dict_keys_subset

def get_hour_from_dict_key(dict_key: str):
    return  int(dict_key[11:13])


def timestamp_str(ts:Timestamp):
    date_time_str =  str(ts).split(' ')
    date_str = date_time_str[0]
    time_str = date_time_str[1] 
    return date_str, time_str


if __name__ == "__main__":

    os.system('clear')
    print("------------------ Running main function for Open Meteo data import -------------------")
    
    time_from = pd.to_datetime("2020-03-22 10:00:00")
    time_to = pd.to_datetime("2020-03-23 12:00:00")

    lon = 16.158
    lat = 58.5812
    [w_data_dict, dict_keys] = get_data_from_api_call(lon, lat, time_from, time_to)

