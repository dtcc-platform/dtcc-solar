import requests
import pandas as pd
import numpy as np
from pandas import Timestamp
from pprint import pp
from dtcc_solar.utils import Sun


def get_data_from_api_call(lon: float, lat: float, suns: list[Sun]):
    date_from_str = timestamp_str(suns[0].datetime_ts)
    date_to_str = timestamp_str(suns[-1].datetime_ts)

    # API call cannot handle time as part of the date variable. So the data is first
    # collected based on the date and then a subset is retriwed which also considers
    # the time in the date variable.
    url_1 = "https://archive-api.open-meteo.com/v1/archive?"
    url_2 = "latitude=" + str(lat) + "&longitude=" + str(lon)
    url_3 = "&start_date=" + date_from_str + "&end_date=" + date_to_str
    url_4 = "&hourly=direct_normal_irradiance"
    url_5 = "&hourly=direct_radiation"  # diffuse_radiation"
    url_6 = "&hourly=diffuse_radiation"

    # Direct normal irradiance, units W/m2
    # Average of the preceding hour on the normal plane (perpendicular to the sun)
    url_ni = url_1 + url_2 + url_3 + url_4
    normal_irradiance = requests.get(url_ni)
    status_ni = normal_irradiance.status_code

    # Direct solar radiation, units W/m2
    # Direct solar radiation as average of the preceding hour
    url_hr = url_1 + url_2 + url_3 + url_5
    direct_radiation = requests.get(url_hr)
    status_dr = direct_radiation.status_code

    # Diffuse solar radiation, units W/m2
    # Diffuse solar radiation as average of the preceding hour
    url_hr = url_1 + url_2 + url_3 + url_6
    diffuse_radiation = requests.get(url_hr)
    status_hr = diffuse_radiation.status_code

    sun_counter = 0

    if status_ni == 200 and status_dr == 200 and status_hr == 200:
        api_dates = normal_irradiance.json()["hourly"]["time"]
        api_dates = format_api_dates(api_dates)

        normal_irradiance_hourly = normal_irradiance.json()["hourly"][
            "direct_normal_irradiance"
        ]
        direct_radiation_hourly = direct_radiation.json()["hourly"]["direct_radiation"]
        diffuse_radiation_hourly = diffuse_radiation.json()["hourly"][
            "diffuse_radiation"
        ]

        for i in range(len(api_dates)):
            if sun_counter < len(suns):
                sun_date = suns[sun_counter].datetime_str
                if date_match(api_dates[i], sun_date):
                    suns[sun_counter].irradiance_dn = normal_irradiance_hourly[i]
                    suns[sun_counter].irradiance_dh = direct_radiation_hourly[i]
                    suns[sun_counter].irradiance_di = diffuse_radiation_hourly[i]
                    sun_counter += 1

        return suns

    elif status_ni == 400 or status_dr == 400:
        print("Open Meteo HTTP status code 400:")
        print(
            "The request points to a resource that do not exist."
            + "This might happen if you query for a station that do"
            + "not produce data for a specific parameter or you are"
            + "using a deprecated version of the API that has been"
            + "removed. It might also be the case that the specified"
            + "station do not have any data for the latest hour."
        )
        return False


# Compare the date stamp from the api data with the date stamp for the generated sun and sky data.
def date_match(api_date, sun_date):
    api_day = api_date[0:10]
    sun_day = sun_date[0:10]
    api_time = api_date[11:19]
    sun_time = api_date[11:19]
    if api_day == sun_day and api_time == sun_time:
        return True
    return False


# The dict key format is (without blank spaces):
# Year - Month - Day T hour : minute : second
# Example 2018-03-23T12:00:00
def format_api_dates(dict_keys: list[str]):
    for i in range(len(dict_keys)):
        dict_keys[i] += ":00"
    return dict_keys


def format_date_range(dict_keys_subset: pd.DatetimeIndex):
    dict_keys_list = []
    for key in dict_keys_subset:
        dict_keys_list.append(key)
    pass


def timestamp_str(ts: Timestamp):
    date_time_str = str(ts).split(" ")
    date_str = date_time_str[0]
    return date_str
