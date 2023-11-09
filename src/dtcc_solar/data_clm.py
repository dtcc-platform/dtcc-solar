import os
import pandas as pd
import numpy as np
from pprint import pp
from dtcc_solar import utils
from dtcc_solar.utils import SunCollection
from dtcc_solar.logging import info, debug, warning, error


# This function reads a *.clm weather file, which contains recorde data for a full year which has been
# compiled by combining data from different months for the years inbetween 2007 and 2021. The time span
# defined in by arguments if then used to obain a sub set of the data for analysis.
def import_data(sunc: SunCollection, weather_file: str):
    name_parts = weather_file.split(".")
    if name_parts[-1] != "clm":
        error(
            "The wrong file type was provided. File extension *.clm was expected, but *."
            + name_parts[-1]
            + " was given."
        )
        return None

    line_index = 0
    data_counter = 0
    year = sunc.time_stamps[0].year
    full_year_start_date = str(year) + "-01-01 00:00:00"
    full_year_end_date = str(year) + "-12-31 23:59:00"

    # The data file contains weather data for an entire year
    year_dates = pd.date_range(
        start=full_year_start_date, end=full_year_end_date, freq="1H"
    )
    year_dict_keys = np.array([str(d) for d in year_dates])
    year_normal_irr = dict.fromkeys(year_dict_keys)
    year_diffuse_irr = dict.fromkeys(year_dict_keys)

    with open(weather_file, "r") as f:
        for line in f:
            # The first 12 lines contains information of the data structure
            if line_index > 12 and line[0] != "*":
                [normal_irr, diffuse_irr] = line2numbers(line)
                year_normal_irr[year_dict_keys[data_counter]] = normal_irr
                year_diffuse_irr[year_dict_keys[data_counter]] = diffuse_irr
                data_counter += 1
            line_index += 1

    all_dates = list(year_normal_irr.keys())
    sun_index = 0

    for clm_date in all_dates:
        if sun_index < sunc.count:
            sun_date = sunc.datetime_strs[sun_index]
            if date_match(clm_date, sun_date):
                sunc.irradiance_dn[sun_index] = year_normal_irr[clm_date]
                sunc.irradiance_di[sun_index] = year_diffuse_irr[clm_date]
                sun_index += 1

    info(f"Weather data successfully collected from: {weather_file}")

    return sunc


def date_match(api_date: str, sun_date: str):
    api_day = api_date[0:10]
    sun_day = sun_date[0:10]
    api_time = api_date[11:19]
    sun_time = sun_date[11:19]
    if api_day == sun_day and api_time == sun_time:
        return True
    return False


def line2numbers(line: str):
    line = line.replace("\n", "")
    linesegs = line.split(",")
    if len(linesegs) == 6:
        normal_irradiance = float(linesegs[2])
        diffuse_horizontal_irradiance = float(linesegs[0])

    return normal_irradiance, diffuse_horizontal_irradiance
