import requests
import os
import time
import pandas as pd
import numpy as np
from pandas import Timestamp
from pprint import pp
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from dtcc_solar import data_io
from dtcc_solar import utils
from dtcc_solar.utils import Sun
from typing import List, Dict
import json

def get_data_from_api_call(lon:float, lat:float, suns:List[Sun]):

    [lon, lat] = check_geo_data(lon, lat)
    strong_data_path = "https://opendata-download-metanalys.smhi.se/api/category/strang1g/version/1/geotype/point/"
    point_data_path_ni = "lon/" + str(lon) + "/lat/" + str(lat) + "/parameter/118/data.json"
    point_data_path_hi = "lon/" + str(lon) + "/lat/" + str(lat) + "/parameter/121/data.json"

    date_from_str = timestamp_str(suns[0].datetime_ts)
    date_to_str = timestamp_str(suns[-1].datetime_ts)

    date_from_to_str = "?from=" + str(date_from_str) + "&to=" + str(date_to_str)

    #Get the normal irradiance
    normal_irradiance =  requests.get(strong_data_path + point_data_path_ni + date_from_to_str)
    status_ni = normal_irradiance.status_code

    #Get the diffuse horozontal irradiance
    horizon_irradiance = requests.get(strong_data_path + point_data_path_hi + date_from_to_str)
    status_hi = horizon_irradiance.status_code 

    if(status_ni == 200 and status_hi == 200):
        ni_json = normal_irradiance.json()
        hi_json = horizon_irradiance.json()

        if len(ni_json) != len(suns):
            print("Missmatch between suns and API data")
            return None

        for i in range(len(suns)):
            api_date = ni_json[i]['date_time']
            sun_date = suns[i].datetime_str
            if date_match(api_date, sun_date):
                suns[i].irradiance_dn = ni_json[i]['value']
                suns[i].irradiance_dh = hi_json[i]['value']
        

        return suns

    
    elif (status_ni == 404 or status_hi == 404):
        print("SMHI Open Data API Docs HTTP code 404:")
        print("The request points to a resource that do not exist." +
        "This might happen if you query for a station that do" +
        "not produce data for a specific parameter or you are" +
        "using a deprecated version of the API that has been" +
        "removed. It might also be the case that the specified" +
        "station do not have any data for the latest hour.")
    
    elif (status_ni == 500 or status_hi == 500):
        print("SMHI Open Data API Docs HTTP code 500:")    
        print("Something went wrong internally in the system. This" +
               "might be fixed after a while so try again later on.")

    return None    

# Compare the date stamp from the api data with the date stamp for the generated suns data.
def date_match(api_date, sun_date):
    api_day = api_date[0:10]
    sun_day = sun_date[0:10]
    api_time = api_date[11:19]
    sun_time = api_date[11:19]

    if api_day == sun_day and api_time == sun_time:
        return True

    return False

def timestamp_str(ts:Timestamp):
    date_time_str =  str(ts).split(' ')
    date = date_time_str[0]
    time = date_time_str[1]
    date_time_str = date + 'T' + time 
    return date_time_str

def make_double_digit_str(s:str):
    if len(s) == 1:
        s = '0' + s
    return s 

def format_dict_key(dict_key:str):
    
    new_dict_key = dict_key[0:19]
    is_format_ok = utils.is_dict_key_format_correct(new_dict_key)
    if(is_format_ok):
        return new_dict_key
        
    return None    


def check_geo_data(lon, lat):

    # Data from the SMHI north of Europe databased is limited to:
    # 1. Records from 2014 and later
    # 2. The domain is limited to: Koordinater för området ges av hörnen:
    #       south west corner: Lon 2.0,  Lat 52.4 
    #       north west corner: Lon 9.5,  Lat 71.5, 
    #       north east corner: Lon 39.7, Lat 71.2
    #       south east corner: Lon 27.8, Lat 52.3 

    test_pt = Point(lon, lat)
    geo_coords = [(2.0, 52.4), (9.5, 71.5), (39.7, 71.2), (27.8, 52.3)]
    geo_polygon = Polygon(geo_coords)

    if(not geo_polygon.contains(test_pt)):
        p1, p2 = nearest_points(geo_polygon, test_pt)
        lon = p1.x
        lat = p1.y
        print("Warning: Location is outside of weather data domain" +
              "The closest location to the domain is locaded at, " +
              "(lon: " + str(lon) + ", lat: " + str(lat) + ")" )

    return lon, lat


def get_shmi_stations_from_api():

    url = "https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/1?measuringStations=all"
    stations = requests.get(url)
    status = stations.status_code
    stations_dict = {}
    
    if(status == 200):
        for line in stations.iter_lines():
            
            if line:
                text_line = str(line)
                        
                if(text_line.find("<id>"))!= -1:
                    segs = text_line.split('.')
                    some_seg = segs[3]
                    segs = some_seg.split('/')
                    id = segs[-1]
                    
                if(text_line.find("Latitud: ")) != -1:
                    segs = text_line.split(':')
                    lat_segs = segs[1].split(' ')
                    lat_num = float(lat_segs[1])

                    lon_segs = segs[2].split(' ')
                    lon_num = float(lon_segs[1])

                    types = ['Latitude', 'Longitude']
                    pos_dict = dict.fromkeys([t for t in types])
                    pos_dict['Latitude'] = lat_num
                    pos_dict['Longitude'] = lon_num
                    stations_dict[id] = pos_dict

        return stations_dict
    
    elif(status == 404):
        print("SMHI Open Data API Docs HTTP code 404:")
        print("The request points to a resource that do not exist." +
        "This might happen if you query for a station that do" +
        "not produce data for a specific parameter or you are" +
        "using a deprecated version of the API that has been" +
        "removed. It might also be the case that the specified" +
        "station do not have any data for the latest hour.")
    elif (status == 500):
        print("SMHI Open Data API Docs HTTP code 500:")    
        print("Something went wrong internally in the system. This" +
               "might be fixed after a while so try again later on.")    

    return None



if __name__ == "__main__":

    os.system('clear')
    print("------------------ Running main function for SMHI API data import -------------------")
    
    time_from_str = "2020-03-22"
    time_to_str = "2020-04-01"
    time_from = pd.to_datetime(time_from_str)
    time_to = pd.to_datetime(time_to_str)

    suns = utils.create_suns(time_from_str, time_to_str)

    lon = 16.158
    lat = 58.5812
    suns = get_data_from_api_call(lon, lat, suns)

    pp(suns)





