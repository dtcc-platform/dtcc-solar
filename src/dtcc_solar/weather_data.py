import os
import numpy as np
import pandas as pd
import math
from dtcc_solar.utils import Parameters, DataSource, Sun
from dtcc_solar import smhi_data, meteo_data, epw_data, clm_data


def append_weather_data(p: Parameters, suns: list[Sun]):
    if p.data_source == DataSource.smhi:
        suns = smhi_data.get_data_from_api_call(p.longitude, p.latitude, suns)
    if p.data_source == DataSource.meteo:
        suns = meteo_data.get_data_from_api_call(p.longitude, p.latitude, suns)
    elif p.data_source == DataSource.clm:
        suns = clm_data.import_weather_data(suns, p.weather_file)
    elif p.data_source == DataSource.epw:
        suns = epw_data.import_weather_data(suns, p.weather_file)
    return suns
