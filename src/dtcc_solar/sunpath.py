import os
import math
import pytz
import datetime
import numpy as np
import pandas as pd
from dtcc_solar import utils
from dtcc_solar.utils import Vec3, Sun
from dtcc_solar.utils import Parameters
from pvlib import solarposition

# from timezonefinder import TimezoneFinder
from pprint import pp


class Sunpath:
    lat: float
    lon: float
    origin: np.ndarray
    radius: float

    def __init__(self, lat: float, lon: float, radius: float):
        self.lat = lat
        self.lon = lon
        self.radius = radius
        self.origin = np.array([0, 0, 0])

    def get_analemmas(self, year: int, sample_rate: int):
        start_date = str(year) + "-01-01 12:00:00"
        end_date = str(year + 1) + "-01-01 11:00:00"
        times = pd.date_range(start=start_date, end=end_date, freq="H")

        # Times to evaluate will be 8760 for a normal year and 8764 for leep year
        times_to_evaluate = len(times.values)

        # Number of days will be 365 or 366 at leep year
        num_of_days = int(np.ceil(times_to_evaluate / 24))

        # Get solar position from the pvLib libra
        sol_pos_hour = solarposition.get_solarposition(times, self.lat, self.lon)

        # Reduce the number of days to reduce the density of the sunpath diagram
        days = np.zeros(num_of_days)
        num_evaluated_days = len(days[0 : len(days) : sample_rate])

        elev = np.zeros((24, num_evaluated_days))
        azim = np.zeros((24, num_evaluated_days))
        zeni = np.zeros((24, num_evaluated_days))

        loop_hours = np.unique(sol_pos_hour.index.hour)
        sun_pos_dict = dict.fromkeys([h for h in loop_hours])

        r = self.radius

        # Get hourly sun path loops in matrix form and elevaion, azimuth and zenith coordinates
        for h in loop_hours:
            subset = sol_pos_hour.loc[sol_pos_hour.index.hour == h, :]
            rad_elev = np.radians(subset.apparent_elevation)
            rad_azim = np.radians(subset.azimuth)
            rad_zeni = np.radians(subset.zenith)
            elev[h, :] = rad_elev.values[0 : len(rad_elev.values) : sample_rate]
            azim[h, :] = rad_azim.values[0 : len(rad_azim.values) : sample_rate]
            zeni[h, :] = rad_zeni.values[0 : len(rad_zeni.values) : sample_rate]

        # Convert hourly sun path loops from spherical to cartesian coordiantes
        for h in range(0, 24):
            x = r * np.cos(elev[h, :]) * np.cos(-azim[h, :]) + self.origin[0]
            y = r * np.cos(elev[h, :]) * np.sin(-azim[h, :]) + self.origin[1]
            z = r * np.sin(elev[h, :]) + self.origin[2]

            sun_pos_dict[h] = utils.create_list_of_vec3(x, y, z)

        return sun_pos_dict

    def get_daypaths(self, dates: pd.DatetimeIndex, minute_step: float):
        n = len(dates.values)
        n_evaluation = int(math.ceil(24 * 60 / minute_step))
        elev = np.zeros((n, n_evaluation + 1))
        azim = np.zeros((n, n_evaluation + 1))
        date_counter = 0

        days_dict = dict.fromkeys([d for d in range(0, n)])
        min_step_str = str(minute_step) + "min"
        day_step = pd.Timedelta(str(24) + "h")

        for date in dates.values:
            times = pd.date_range(date, date + day_step, freq=min_step_str)
            sol_pos_day = solarposition.get_solarposition(times, self.lat, self.lon)
            rad_elev_day = np.radians(sol_pos_day.apparent_elevation)
            rad_azim_day = np.radians(sol_pos_day.azimuth)
            elev[date_counter, :] = rad_elev_day.values
            azim[date_counter, :] = rad_azim_day.values
            date_counter = date_counter + 1

        for d in range(0, date_counter):
            x = self.radius * np.cos(elev[d, :]) * np.cos(-azim[d, :]) + self.origin[0]
            y = self.radius * np.cos(elev[d, :]) * np.sin(-azim[d, :]) + self.origin[1]
            z = self.radius * np.sin(elev[d, :]) + self.origin[2]

            days_dict[d] = utils.create_list_of_vec3(x, y, z)

        return days_dict

    def get_suns_positions(self, suns: list[Sun]):
        date_from_str = suns[0].datetime_str
        date_to_str = suns[-1].datetime_str
        dates = pd.date_range(start=date_from_str, end=date_to_str, freq="1H")

        solpos = solarposition.get_solarposition(dates, self.lat, self.lon)
        elev = np.radians(solpos.apparent_elevation.to_list())
        azim = np.radians(solpos.azimuth.to_list())
        zeni = np.radians(solpos.zenith.to_list())
        x_sun = self.radius * np.cos(elev) * np.cos(-azim) + self.origin[0]
        y_sun = self.radius * np.cos(elev) * np.sin(-azim) + self.origin[1]
        z_sun = self.radius * np.sin(elev) + self.origin[2]

        if len(suns) == len(dates):
            for i in range(len(suns)):
                suns[i].position = Vec3(x=x_sun[i], y=y_sun[i], z=z_sun[i])
                suns[i].zenith = zeni[i]
                suns[i].over_horizon = zeni[i] < math.pi / 2
                suns[i].sun_vec = utils.normalise_vector3(
                    Vec3(
                        x=self.origin[0] - x_sun[i],
                        y=self.origin[1] - y_sun[i],
                        z=self.origin[2] - z_sun[i],
                    )
                )
        else:
            print("Something went wrong in when retrieving solar positions!")

        return suns

    def create_sun_timestamps(self, start_date: str, end_date: str):
        time_from = pd.to_datetime(start_date)
        time_to = pd.to_datetime(end_date)
        suns = []
        index = 0
        times = pd.date_range(start=time_from, end=time_to, freq="H")
        for time in times:
            sun = Sun(str(time), time, index)
            suns.append(sun)
            index += 1

        return suns

    def create_suns(self, p: Parameters):
        suns = self.create_sun_timestamps(p.start_date, p.end_date)
        suns = self.get_suns_positions(suns)
        return suns


class SunpathUtils:
    def __init__(self):
        pass

    @staticmethod
    def convert_utc_to_local_time(utc_h, gmt_diff):
        local_h = utc_h + gmt_diff
        if local_h < 0:
            local_h = 24 + local_h
        elif local_h > 23:
            local_h = local_h - 24

        return local_h

    @staticmethod
    def convert_local_time_to_utc(local_h, gmt_diff):
        utc_h = local_h - gmt_diff
        if utc_h < 0:
            utc_h = 24 + utc_h
        elif utc_h > 23:
            utc_h = utc_h - 24

        return utc_h

    @staticmethod
    def get_timezone_from_long_lat(lat, long):
        tf = 0  # = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=long, lat=lat)
        print(timezone_str)

        timezone = pytz.timezone(timezone_str)
        dt = datetime.datetime.now()
        offset = timezone.utcoffset(dt)
        h_offset_1 = offset.seconds / 3600
        h_offset_2 = 24 - h_offset_1

        print("Time zone: " + str(timezone))
        print("GMT_diff: " + str(h_offset_1) + " or: " + str(h_offset_2))

        h_offset = np.min([h_offset_1, h_offset_2])

        return h_offset


if __name__ == "__main__":
    pass
