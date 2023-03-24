import requests
import os
import pandas as pd
from pandas import Timestamp
from pprint import pp
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points


def get_smhi_data_from_api_call(lon:float, lat:float, date_from:Timestamp, date_to:Timestamp):

    [lon, lat] = check_geo_data(lon, lat)
    strong_data_path = "https://opendata-download-metanalys.smhi.se/api/category/strang1g/version/1/geotype/point/"
    point_data_path_ni = "lon/" + str(lon) + "/lat/" + str(lat) + "/parameter/118/data.json"
    point_data_path_hi = "lon/" + str(lon) + "/lat/" + str(lat) + "/parameter/121/data.json"

    date_from_str = get_datetime_in_api_format_from_timestamp(date_from)
    date_to_str = get_datetime_in_api_format_from_timestamp(date_to)

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
        #Store the data in a dictionary
        keys = []
        for i in range(len(ni_json)):
            key = ni_json[i]['date_time']
            dt = pd.to_datetime(key)
            keys.append(key)

        w_data_dict = dict.fromkeys(keys)
        for i, key in enumerate(keys):
            ni = ni_json[i]['value']
            hi = hi_json[i]['value']
            types = ['normal_irradiance', 'horizontal_irradiance']
            data_dict = dict.fromkeys([t for t in types])
            data_dict['normal_irradiance'] = ni
            data_dict['horizontal_irradiance'] = hi
            w_data_dict[key] = data_dict
        
        return w_data_dict

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

def get_datetime_in_api_format_from_timestamp(ts:Timestamp):
    date_time_str =  str(ts).split(' ')
    date = date_time_str[0]
    time = date_time_str[1]
    date_time_str = date + 'T' + time 
    return date_time_str

def make_double_digit_str(s:str):
    if len(s) == 1:
        s = '0' + s
    return s    

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

# Todo: Move to unit test file.
def assert_no_summer_time_in_data():

    # In order to test the the SMHI data does not acount for summer time shift, (which 
    # need to be sycronised with the solar position calulations), the hour 02:00 should
    # present in the weather data in the last sunday of march and in the last sunday of
    # october. This structure is based on an EU regulation but may not be followed by 
    # countries outside of EU. However, the SMHI Open data API only covers the north of 
    # Europe so the check is still relevant for this application.    

    # Checking that 02:00:00 existing in the transion from winter to summer.
    dst_test_from = ["2018-03-25 00:00:00", "2018-10-28 00:00:00", 
                     "2019-03-31 00:00:00", "2019-10-27 00:00:00", 
                     "2020-03-29 00:00:00", "2020-10-25 00:00:00",
                     "2021-03-28 00:00:00", "2021-10-31 00:00:00",
                     "2022-03-27 00:00:00", "2022-10-30 00:00:00"]
    
    dst_test_to =   ["2018-03-25 04:00:00", "2018-10-28 04:00:00", 
                     "2019-03-31 04:00:00", "2019-10-27 04:00:00", 
                     "2020-03-29 04:00:00", "2020-10-25 04:00:00",
                     "2021-03-28 04:00:00", "2021-10-31 04:00:00",
                     "2022-03-27 04:00:00", "2022-10-30 04:00:00"]
    
    lon = 16.158
    lat = 58.5812
    results = []
    for i in range(0, len(dst_test_from)):
        time_from = dst_test_from[i]
        time_to =   dst_test_to[i]
        w_data_dict = get_smhi_data_from_api_call(lon, lat, time_from, time_to)
        for key in w_data_dict:
            parts = key.split('T')
            part2 = parts[1]
            hour = part2[0:2]
            if(hour == "02"):
                results.append(True)
                break
            
    if(all(results)):                
        return True
    
    return False


if __name__ == "__main__":

    os.system('clear')
    print("------------------ SMHI API test -------------------")
    
    time_from = pd.to_datetime("2020-03-22")
    time_to = pd.to_datetime("2020-04-01")
    lon = 16.158
    lat = 58.5812
    w_data_dict = get_smhi_data_from_api_call(lon, lat, time_from, time_to)

    pp(w_data_dict)

    result = assert_no_summer_time_in_data()

    print(result)


