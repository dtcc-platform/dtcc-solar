import os
import numpy as np
import pandas as pd
import math
from dtcc_solar.utils import Parameters, DataSource, Sun
from dtcc_solar import data_clm, data_epw, data_meteo, data_smhi


def append_weather_data(p: Parameters, suns: list[Sun]):
    if p.data_source == DataSource.smhi:
        suns = data_smhi.get_data(p.longitude, p.latitude, suns)
    if p.data_source == DataSource.meteo:
        suns = data_meteo.get_data(p.longitude, p.latitude, suns)
    elif p.data_source == DataSource.clm:
        suns = data_clm.import_data(suns, p.weather_file)
    elif p.data_source == DataSource.epw:
        suns = data_epw.import_data(suns, p.weather_file)
    return suns
