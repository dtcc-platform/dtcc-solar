import os
import math
import pytz
import datetime
import numpy as np
import pandas as pd
from dtcc_solar import utils
from dtcc_solar.utils import SunCollection, DataSource, unitize, SunApprox
from dtcc_solar.utils import SolarParameters
from pvlib import solarposition
from dtcc_solar import data_clm, data_epw, data_meteo, data_smhi
from dtcc_model import Mesh, PointCloud
from dtcc_solar.colors import get_blended_color_yellow_red
from dtcc_solar.utils import concatenate_meshes
from dtcc_solar.logging import info, debug, warning, error
from dtcc_solar.sungroups import SunGroups
from dtcc_solar.sundome import SunDome

# from timezonefinder import TimezoneFinder
from pprint import pp


class Sunpath:
    lat: float
    lon: float
    origin: np.ndarray
    r: float  # Radius of sunpath diagram
    w: float  # Width of paths in sunpath diagram

    sunc: SunCollection
    origin: np.ndarray
    sun_pc: list[PointCloud]
    all_suns_pc: PointCloud
    mesh: Mesh  # Combination of annalemmas and day paths
    analemmas_meshes: list[Mesh]  # Analemmas for each hour in a year
    daypath_meshes: list[Mesh]  # Day paths for three dates in a year
    analemmas_pc: PointCloud  # Analemmas for each hour in a year as a point cloud

    sungroups: SunGroups
    sundome: SunDome

    def __init__(self, p: SolarParameters, radius: float, include_night: bool = False):
        self.lat = p.latitude
        self.lon = p.longitude
        self.r = radius
        self.origin = np.array([0, 0, 0])
        self.sungroups = None
        self.sundome = None
        self.create_suns(p, include_night)

        if p.display:
            self._build_sunpath_mesh()

        if p.sun_approx == SunApprox.group:
            sun_pos_dict = self.get_analemmas(2019, p.suns_per_group)
            self.sungroups = SunGroups(sun_pos_dict, self.sunc)
        elif p.sun_approx == SunApprox.quad:
            dates = pd.date_range(start="2019-01-01", end="2019-12-31", freq="1D")
            sun_pos_dict = self.get_daypaths(dates, 10)
            self.sundome = SunDome(sun_pos_dict, self.r, p.sundome_div)

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

        r = self.r

        # Get hourly sun path loops in matrix form and elevaion, azimuth and zenith coordinates
        for h in loop_hours:
            subset = sol_pos_hour.loc[sol_pos_hour.index.hour == h, :]
            rad_elev = np.radians(subset.elevation)
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
            rad_elev_day = np.radians(sol_pos_day.elevation)
            rad_azim_day = np.radians(sol_pos_day.azimuth)
            elev[date_counter, :] = rad_elev_day.values
            azim[date_counter, :] = rad_azim_day.values
            date_counter = date_counter + 1

        for d in range(0, date_counter):
            x = self.r * np.cos(elev[d, :]) * np.cos(-azim[d, :]) + self.origin[0]
            y = self.r * np.cos(elev[d, :]) * np.sin(-azim[d, :]) + self.origin[1]
            z = self.r * np.sin(elev[d, :]) + self.origin[2]

            days_dict[d] = utils.create_list_of_vec3(x, y, z)

        return days_dict

    def create_suns(self, p: SolarParameters, include_night: bool = False):
        over_horizon = []
        self.sunc = SunCollection()
        self._create_sun_timestamps(p.start_date, p.end_date)
        self._get_suns_positions(over_horizon)
        self._append_weather_data(p)
        if not include_night:
            self._remove_suns_below_horizon(over_horizon)

    def _create_sun_timestamps(self, start_date: str, end_date: str):
        time_from = pd.to_datetime(start_date)
        time_to = pd.to_datetime(end_date)
        times = pd.date_range(start=time_from, end=time_to, freq="H")
        self.sunc.count = len(times)
        for i, time in enumerate(times):
            time_stamp = pd.Timestamp(time)
            self.sunc.time_stamps.append(time_stamp)
            self.sunc.datetime_strs.append(str(time_stamp))

    def _get_suns_positions(self, over_horizon: list[bool]):
        date_from_str = self.sunc.datetime_strs[0]
        date_to_str = self.sunc.datetime_strs[-1]
        dates = pd.date_range(start=date_from_str, end=date_to_str, freq="1H")
        self.sunc.date_times = dates
        solpos = solarposition.get_solarposition(dates, self.lat, self.lon)
        elev = np.radians(solpos.elevation.to_list())
        azim = np.radians(solpos.azimuth.to_list())
        zeni = np.radians(solpos.zenith.to_list())
        x_sun = self.r * np.cos(elev) * np.cos(-azim) + self.origin[0]
        y_sun = self.r * np.cos(elev) * np.sin(-azim) + self.origin[1]
        z_sun = self.r * np.sin(elev) + self.origin[2]
        sun_vecs = []
        positions = []
        zeniths = []
        for i in range(self.sunc.count):
            sun_vec = unitize(np.array([x_sun[i], y_sun[i], z_sun[i]]))
            # length = np.linalg.norm(sun_vec)
            # print(length)
            sun_vecs.append(sun_vec)
            positions.append(np.array([x_sun[i], y_sun[i], z_sun[i]]))
            zeniths.append(zeni[i])
            over_horizon.append(zeni[i] < math.pi / 2)

        self.sunc.sun_vecs = np.array(sun_vecs)
        self.sunc.positions = np.array(positions)
        self.sunc.zeniths = np.array(zeniths)
        self.sunc.irradiance_dh = np.zeros(self.sunc.count)
        self.sunc.irradiance_di = np.zeros(self.sunc.count)
        self.sunc.irradiance_dn = np.zeros(self.sunc.count)

    def _remove_suns_below_horizon(self, over_horizon: list[bool]):
        # Remove data where the sun is below the horizon
        self.sunc.sun_vecs = self.sunc.sun_vecs[over_horizon, :]
        self.sunc.positions = self.sunc.positions[over_horizon, :]
        self.sunc.date_times = self.sunc.date_times[over_horizon]
        self.sunc.irradiance_dh = self.sunc.irradiance_dh[over_horizon]
        self.sunc.irradiance_di = self.sunc.irradiance_di[over_horizon]
        self.sunc.irradiance_dn = self.sunc.irradiance_dn[over_horizon]
        self.sunc.count = len(self.sunc.positions)

        timestamps = []
        datestrings = []
        for i, ts in enumerate(self.sunc.time_stamps):
            if over_horizon[i]:
                timestamps.append(ts)
                datestrings.append(self.sunc.datetime_strs[i])

        self.sunc.time_stamps = timestamps
        self.sunc.datetime_strs = datestrings

    def _append_weather_data(self, p: SolarParameters):
        if p.data_source == DataSource.smhi:
            data_smhi.get_data(p.longitude, p.latitude, self.sunc)
        if p.data_source == DataSource.meteo:
            data_meteo.get_data(p.longitude, p.latitude, self.sunc)
        elif p.data_source == DataSource.clm:
            data_clm.import_data(self.sunc, p.weather_file)
        elif p.data_source == DataSource.epw:
            data_epw.import_data(self.sunc, p.weather_file)

    def _build_sunpath_mesh(self):
        self.w = self.r / 300

        # Get analemmas mesh and sun positions represented as a point cloud
        sun_pos_dict = self.get_analemmas(2019, 2)
        self.analemmas_meshes = self.create_mesh_strip(sun_pos_dict, self.r, self.w)
        self.analemmas_pc = self.create_sunpath_pc(sun_pos_dict)

        # Get every sun position for each minute in a year as points in a point cloud
        dates = pd.date_range(start="2019-01-01", end="2019-12-31", freq="1D")
        sun_pos_dict = self.get_daypaths(dates, 1)
        self.all_suns_pc = self.create_sunpath_pc(sun_pos_dict)

        # Get day path loops as mesh strips
        days = pd.to_datetime(["2019-06-21", "2019-03-21", "2019-12-21"])
        days_dict = self.get_daypaths(days, 2)
        self.daypath_meshes = self.create_mesh_strip(days_dict, self.r, self.w)

        # Create pc representation of current sun positions
        self.sun_pc = self._create_sun_spheres()

        mesh1 = concatenate_meshes(self.analemmas_meshes)
        mesh2 = concatenate_meshes(self.daypath_meshes)
        self.mesh = concatenate_meshes([mesh1, mesh2])

        info("Sunpath mesh representation created")

    def create_mesh_strip(self, sun_pos_dict, radius: float, width: float):
        meshes = []

        for h in sun_pos_dict:
            n_suns = len(sun_pos_dict[h])
            pos = sun_pos_dict[h]
            offset_vertices_L = np.zeros((n_suns + 1, 3))
            offset_vertices_R = np.zeros((n_suns + 1, 3))

            v_counter = 0
            vertices = []
            faces = []
            vertex_colors = []

            for i in range(0, n_suns - 1):
                i_next = i + 1
                if i == n_suns - 1:
                    i_next = 0

                sun_pos_1 = np.array([pos[i].x, pos[i].y, pos[i].z])
                sun_pos_2 = np.array([pos[i_next].x, pos[i_next].y, pos[i_next].z])

                vec_1 = unitize(0.5 * (sun_pos_1 + sun_pos_2))
                vec_2 = unitize(sun_pos_2 - sun_pos_1)
                vec_3 = np.cross(vec_1, vec_2)

                # Offset vertices to create a width for path
                offset_vertices_L[i, :] = sun_pos_1 + (width * vec_3)
                offset_vertices_R[i, :] = sun_pos_1 - (width * vec_3)

                vertices.append(offset_vertices_L[i, :])
                vertices.append(offset_vertices_R[i, :])

                if i < (n_suns - 2):
                    faces.append([v_counter, v_counter + 2, v_counter + 1])
                    faces.append([v_counter + 1, v_counter + 2, v_counter + 3])
                    v_counter += 2
                else:
                    faces.append([v_counter, 0, v_counter + 1])
                    faces.append([v_counter + 1, 0, 1])

                v_color = get_blended_color_yellow_red(radius, pos[i].z)

                vertex_colors.append(v_color)
                vertex_colors.append(v_color)

            vertices = np.array(vertices)
            faces = np.array(faces)
            vertex_colors = np.array(vertex_colors)
            mesh = Mesh(vertices=vertices, vertex_colors=vertex_colors, faces=faces)
            meshes.append(mesh)

        return meshes

    def _create_sun_spheres(self):
        points = []
        points = self.sunc.positions
        points = np.array(points)
        pc = PointCloud(points=points)
        return pc

    def create_sunpath_pc(self, sun_pos_dict):
        points = []
        for h in sun_pos_dict:
            pos_list = sun_pos_dict[h]
            for pos in pos_list:
                sun_pos = np.array([pos.x, pos.y, pos.z])
                points.append(sun_pos)

        points = np.array(points)
        pc = PointCloud(points=points)
        return pc


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
